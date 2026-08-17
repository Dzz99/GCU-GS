from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="cluster_uf",
    version="0.1.0",
    packages=["cluster_uf"],
    ext_modules=[
        CUDAExtension(
            name="cluster_uf._C",
            sources=["ext.cpp", "union_find.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
