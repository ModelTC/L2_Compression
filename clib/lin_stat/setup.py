from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='LinStat',
    ext_modules=[
        CUDAExtension('LinStat', [
            "/".join(__file__.split('/')[:-1] + ['lin_stat.cpp']),
            "/".join(__file__.split('/')[:-1] + ['lin_stat.cu']),
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })