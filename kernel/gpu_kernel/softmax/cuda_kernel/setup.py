from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="kc_softmax_ext",
    ext_modules=[
        CUDAExtension(
            name="kc_softmax_ext",
            sources=["binding.cpp", "kernel.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)

