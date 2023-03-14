from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="ISBNET_OP",
    ext_modules=[
        CUDAExtension(
            "ISBNET_OP",
            ["src/isbnet_api.cpp", "src/isbnet_ops.cpp", "src/cuda.cu"],
            extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
