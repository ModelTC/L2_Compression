from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='TriStat',
    ext_modules=[
        CUDAExtension('TriStat', [
            "/".join(__file__.split('/')[:-1] + ['tri_stat.cpp']),
            "/".join(__file__.split('/')[:-1] + ['tri_stat.cu']),
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })