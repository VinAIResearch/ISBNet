from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension


if __name__ == "__main__":
    setup(
        name="isbnet",
        version="1.0",
        description="isbnet",
        author="Tuan Ngo",
        packages=["isbnet"],
        package_data={"isbnet.ops": ["*/*.so"]},
        ext_modules=[
            CUDAExtension(
                name="isbnet.ops.ops",
                sources=[
                    "isbnet/ops/src/isbnet_api.cpp",
                    "isbnet/ops/src/isbnet_ops.cpp",
                    "isbnet/ops/src/cuda.cu",
                ],
                extra_compile_args={"cxx": ["-g"], "nvcc": ["-O2"]},
            )
        ],
        cmdclass={"build_ext": BuildExtension},
    )
