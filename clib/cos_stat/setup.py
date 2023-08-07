from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='CosStat',
    ext_modules=[
        CUDAExtension('CosStat', [
            "/".join(__file__.split('/')[:-1] + ['cos_stat.cpp']),
            "/".join(__file__.split('/')[:-1] + ['cos_stat.cu']),
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })