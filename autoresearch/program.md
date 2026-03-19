# Unified Autoresearch Program (All Ops, CPU + GPU)

## Objective

Build a continuous autoresearch loop for all operators on both CPU and GPU.
The system optimizes at operator level, never at model graph level.

## Execution Unit

Each task is:

`task = op_id × platform × dtype × shape_bucket`

## X / E / D / L

1. X (Mutable): candidate params or candidate implementation for a single task.
2. E (Evaluator): fixed correctness + performance harness.
3. D (Decision): keep/revert policy based on correctness first, then score gain.
4. L (Logbook): full per-iteration records and portfolio dashboard.

## Loop

1. Build baseline for each task.
2. Iterate candidate proposals.
3. Evaluate with correctness gates.
4. Compute score against baseline.
5. keep/revert.
6. Write timeline + artifacts.
7. Move to next task based on scheduler priority.

## Hard Rules

1. Never keep incorrect candidate.
2. Evaluator harness is immutable during a run.
3. Every attempt must be logged, including failures.
4. Final output must include reproducible commands and best candidate artifacts.

