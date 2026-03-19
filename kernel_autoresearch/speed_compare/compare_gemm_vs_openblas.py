from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernel_autoresearch.harness.bench import candidate_signature
from kernel_autoresearch.python.runner_client import build_cpp_runner
from kernel_autoresearch.python.scheduler import bucket_for_shape
from kernel_autoresearch.python.utils import load_yaml


@dataclass(frozen=True)
class CompareCase:
    name: str
    input_dtype: str
    output_dtype: str
    workspace_rel: str


DEFAULT_CASES: list[CompareCase] = [
    CompareCase(name="f32_default", input_dtype="f32", output_dtype="f32", workspace_rel="kernel_autoresearch/workspace"),
    CompareCase(name="f16_quick", input_dtype="f16", output_dtype="f16", workspace_rel="kernel_autoresearch/workspace_fp16_quick"),
    CompareCase(name="i8_quick", input_dtype="i8", output_dtype="i32", workspace_rel="kernel_autoresearch/workspace_int8_quick"),
]


def repo_root() -> Path:
    return ROOT


def run_cmd(args: list[str], cwd: Path) -> None:
    subprocess.run(args, cwd=str(cwd), check=True)


def ensure_openblas(openblas_root: Path, jobs: int) -> Path:
    static_lib = openblas_root / "libopenblas.a"
    if static_lib.exists():
        return static_lib
    run_cmd(["make", "-C", str(openblas_root), f"-j{jobs}", "NOFORTRAN=1"], cwd=repo_root())
    if not static_lib.exists():
        raise FileNotFoundError(f"OpenBLAS static library not found after build: {static_lib}")
    return static_lib


def ensure_openblas_runner(speed_compare_dir: Path, openblas_root: Path, static_lib: Path) -> Path:
    cpp = speed_compare_dir / "openblas_runner.cc"
    bin_path = speed_compare_dir / "openblas_runner"
    cmd = [
        "g++",
        "-O3",
        "-std=c++17",
        str(cpp),
        "-I",
        str(openblas_root),
        str(static_lib),
        "-lpthread",
        "-lm",
        "-ldl",
        "-o",
        str(bin_path),
    ]
    run_cmd(cmd, cwd=repo_root())
    return bin_path


def parse_last_json(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"cannot parse json from stdout: {stdout[:400]}")


def run_kc(
    runner: Path,
    shape: dict[str, int],
    cand: dict[str, Any],
    input_dtype: str,
    output_dtype: str,
    warmup: int,
    iters: int,
    timeout_sec: int,
) -> dict[str, Any]:
    args = [
        str(runner),
        "--m",
        str(shape["m"]),
        "--n",
        str(shape["n"]),
        "--k",
        str(shape["k"]),
        "--kernel_variant",
        str(cand["kernel_variant"]),
        "--bm",
        str(cand["bm"]),
        "--bn",
        str(cand["bn"]),
        "--bk",
        str(cand["bk"]),
        "--pack_a",
        "1" if cand["pack_a"] else "0",
        "--pack_b",
        "1" if cand["pack_b"] else "0",
        "--simd",
        "1" if cand["simd"] else "0",
        "--threads",
        str(cand["threads"]),
        "--unroll_k",
        str(cand["unroll_k"]),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--verify",
        "1",
        "--json",
        "1",
        "--input_mode",
        "random",
        "--input_dtype",
        input_dtype,
        "--output_dtype",
        output_dtype,
    ]
    cp = subprocess.run(
        args,
        cwd=str(repo_root()),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    if cp.returncode != 0:
        raise RuntimeError(f"kc runner failed rc={cp.returncode}: {cp.stderr.strip()[:300]}")
    row = parse_last_json(cp.stdout)
    row["engine"] = "kernel_autoresearch"
    return row


def run_openblas(
    runner: Path,
    shape: dict[str, int],
    input_dtype: str,
    threads: int,
    warmup: int,
    iters: int,
    timeout_sec: int,
) -> dict[str, Any]:
    args = [
        str(runner),
        "--m",
        str(shape["m"]),
        "--n",
        str(shape["n"]),
        "--k",
        str(shape["k"]),
        "--threads",
        str(threads),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
        "--input_mode",
        "random",
        "--input_dtype",
        input_dtype,
    ]
    cp = subprocess.run(
        args,
        cwd=str(repo_root()),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    if cp.returncode != 0:
        raise RuntimeError(f"openblas runner failed rc={cp.returncode}: {cp.stderr.strip()[:300]}")
    return parse_last_json(cp.stdout)


def load_best_candidates(state_path: Path) -> dict[str, dict[str, Any]]:
    state = json.loads(state_path.read_text(encoding="utf-8"))
    best = state.get("best_candidate_by_bucket", {})
    if not best:
        raise ValueError(f"best_candidate_by_bucket missing in {state_path}")
    return best


def parse_case_spec(spec: str) -> CompareCase:
    # format: name:input_dtype:output_dtype:workspace_rel
    parts = spec.split(":", maxsplit=3)
    if len(parts) != 4:
        raise ValueError(f"invalid --case spec={spec!r}, expect name:input_dtype:output_dtype:workspace_rel")
    return CompareCase(name=parts[0], input_dtype=parts[1], output_dtype=parts[2], workspace_rel=parts[3])


def slower_metrics(kc_lat: float, ob_lat: float) -> tuple[str, float, float]:
    if kc_lat >= ob_lat:
        factor = kc_lat / max(ob_lat, 1e-12)
        return "kernel_autoresearch", factor, (factor - 1.0) * 100.0
    factor = ob_lat / max(kc_lat, 1e-12)
    return "openblas", factor, (factor - 1.0) * 100.0


def write_outputs(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = out_dir / "compare.tsv"
    md_path = out_dir / "report.md"
    json_path = out_dir / "summary.json"

    cols = [
        "case",
        "shape",
        "bucket",
        "input_dtype",
        "output_dtype",
        "openblas_baseline_mode",
        "m",
        "n",
        "k",
        "threads",
        "candidate_signature",
        "kc_latency_ms_p50",
        "kc_gflops",
        "openblas_latency_ms_p50",
        "openblas_gflops",
        "kc_gflops_vs_openblas",
        "kc_latency_speedup_vs_openblas",
        "slower_engine",
        "slower_factor",
        "slower_pct",
    ]
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})

    by_case: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_case.setdefault(str(r["case"]), []).append(r)

    summary_rows: list[dict[str, Any]] = []
    for case_name, rs in by_case.items():
        mean_ratio = sum(float(x["kc_gflops_vs_openblas"]) for x in rs) / len(rs)
        mean_kc_lat = sum(float(x["kc_latency_ms_p50"]) for x in rs) / len(rs)
        mean_ob_lat = sum(float(x["openblas_latency_ms_p50"]) for x in rs) / len(rs)
        slower, factor, pct = slower_metrics(mean_kc_lat, mean_ob_lat)
        summary_rows.append(
            {
                "case": case_name,
                "input_dtype": rs[0]["input_dtype"],
                "output_dtype": rs[0]["output_dtype"],
                "openblas_baseline_mode": rs[0]["openblas_baseline_mode"],
                "mean_kc_gflops_vs_openblas": mean_ratio,
                "mean_kc_latency_ms_p50": mean_kc_lat,
                "mean_openblas_latency_ms_p50": mean_ob_lat,
                "mean_slower_engine": slower,
                "mean_slower_factor": factor,
                "mean_slower_pct": pct,
                "shapes": len(rs),
            }
        )
    summary_rows.sort(key=lambda x: str(x["case"]))
    json_path.write_text(json.dumps({"summary": summary_rows, "rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# GEMM Speed Compare: kernel_autoresearch vs OpenBLAS",
        "",
        f"- Rows: {len(rows)}",
        "",
        "## Mean Summary By Case",
        "",
        "| case | dtype | openblas baseline | mean kc/openblas (GFLOPS) | mean slower side | mean slowdown |",
        "|---|---|---|---:|---|---:|",
    ]
    for s in summary_rows:
        lines.append(
            f"| {s['case']} | {s['input_dtype']}->{s['output_dtype']} | {s['openblas_baseline_mode']} | "
            f"{float(s['mean_kc_gflops_vs_openblas']):.4f} | {s['mean_slower_engine']} | {float(s['mean_slower_pct']):.2f}% |"
        )

    for case_name in sorted(by_case.keys()):
        rs = by_case[case_name]
        lines.extend(
            [
                "",
                f"## Detail: {case_name} ({rs[0]['input_dtype']}->{rs[0]['output_dtype']})",
                "",
                "| shape | bucket | threads | kc_gflops | openblas_gflops | kc/openblas | slower side | slowdown |",
                "|---|---:|---:|---:|---:|---:|---|---:|",
            ]
        )
        for r in rs:
            lines.append(
                f"| {r['shape']} | {r['bucket']} | {int(r['threads'])} | {float(r['kc_gflops']):.3f} | "
                f"{float(r['openblas_gflops']):.3f} | {float(r['kc_gflops_vs_openblas']):.4f} | "
                f"{r['slower_engine']} | {float(r['slower_pct']):.2f}% |"
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"summary_rows": summary_rows, "rows": rows}


def main() -> None:
    root = repo_root()
    speed_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Compare kernel_autoresearch GEMM vs OpenBLAS SGEMM for multiple dtypes")
    parser.add_argument("--config", type=Path, default=root / "kernel_autoresearch/configs/default.yaml")
    parser.add_argument("--openblas-root", type=Path, default=root / "3rdparty/OpenBLAS")
    parser.add_argument("--threads", type=int, default=-1, help="OpenBLAS threads. <=0 means follow KC candidate threads per shape.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=speed_dir / "results_all")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case spec: name:input_dtype:output_dtype:workspace_rel. Repeatable. Default runs f32/f16/i8 preset cases.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    buckets_cfg = cfg["experiment"]["buckets"]
    shapes_cfg = load_yaml(root / cfg["paths"]["shapes_config"])
    shapes = shapes_cfg.get("shapes", [])
    state_file_name = cfg["autoresearch"]["state_file"]

    cases = [parse_case_spec(x) for x in args.case] if args.case else DEFAULT_CASES

    static_lib = ensure_openblas(args.openblas_root, args.jobs)
    openblas_runner = ensure_openblas_runner(speed_dir, args.openblas_root, static_lib)

    cpp_cfg = cfg["paths"]["cpp_runner"]
    kc_runner = build_cpp_runner(
        repo_root=root,
        source_dir=root / cpp_cfg["source_dir"],
        build_dir=root / cpp_cfg["build_dir"],
        build_type=cpp_cfg["cmake_build_type"],
        binary_name=cpp_cfg["binary_name"],
    )

    rows: list[dict[str, Any]] = []
    for case in cases:
        workspace = root / case.workspace_rel
        state_path = workspace / state_file_name
        if not state_path.exists():
            raise FileNotFoundError(f"state file not found for case={case.name}: {state_path}")
        best_by_bucket = load_best_candidates(state_path)

        for shape in shapes:
            bucket = bucket_for_shape(shape, buckets_cfg)
            cand = best_by_bucket.get(bucket)
            if cand is None:
                continue
            kc_input_dtype = str(cand.get("input_dtype", case.input_dtype))
            kc_output_dtype = str(cand.get("output_dtype", case.output_dtype))
            if kc_input_dtype != case.input_dtype or kc_output_dtype != case.output_dtype:
                raise ValueError(
                    f"case={case.name} expects dtype {case.input_dtype}->{case.output_dtype}, "
                    f"but candidate bucket={bucket} has {kc_input_dtype}->{kc_output_dtype}"
                )

            kc = run_kc(
                kc_runner,
                shape,
                cand,
                input_dtype=kc_input_dtype,
                output_dtype=kc_output_dtype,
                warmup=args.warmup,
                iters=args.iters,
                timeout_sec=args.timeout_sec,
            )
            ob_threads = int(cand["threads"]) if args.threads <= 0 else args.threads
            ob = run_openblas(
                openblas_runner,
                shape,
                input_dtype=kc_input_dtype,
                threads=ob_threads,
                warmup=args.warmup,
                iters=args.iters,
                timeout_sec=args.timeout_sec,
            )
            if not bool(kc.get("valid", False)):
                raise RuntimeError(f"KC run invalid for case={case.name} shape={shape['name']}: {kc}")
            if not bool(ob.get("valid", False)):
                raise RuntimeError(f"OpenBLAS run invalid for case={case.name} shape={shape['name']}: {ob}")

            kc_lat = float(kc["latency_ms_p50"])
            ob_lat = float(ob["latency_ms_p50"])
            slower_engine, slower_factor, slower_pct = slower_metrics(kc_lat, ob_lat)

            row = {
                "case": case.name,
                "shape": shape["name"],
                "bucket": bucket,
                "input_dtype": kc_input_dtype,
                "output_dtype": kc_output_dtype,
                "openblas_baseline_mode": str(ob.get("baseline_mode", "native_f32")),
                "m": shape["m"],
                "n": shape["n"],
                "k": shape["k"],
                "threads": ob_threads,
                "candidate_signature": candidate_signature(cand),
                "kc_latency_ms_p50": kc_lat,
                "kc_gflops": float(kc["gflops"]),
                "openblas_latency_ms_p50": ob_lat,
                "openblas_gflops": float(ob["gflops"]),
                "slower_engine": slower_engine,
                "slower_factor": slower_factor,
                "slower_pct": slower_pct,
            }
            row["kc_gflops_vs_openblas"] = row["kc_gflops"] / max(row["openblas_gflops"], 1e-9)
            row["kc_latency_speedup_vs_openblas"] = row["openblas_latency_ms_p50"] / max(row["kc_latency_ms_p50"], 1e-9)
            rows.append(row)

    if not rows:
        raise RuntimeError("No comparable shapes/candidates found")

    out = write_outputs(rows, args.out_dir)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "cases": sorted({x["case"] for x in rows}),
                "out_dir": str(args.out_dir),
                "summary": out["summary_rows"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
