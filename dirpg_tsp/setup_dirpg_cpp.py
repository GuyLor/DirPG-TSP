from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='dirpg_cpp',
      ext_modules=[cpp_extension.CppExtension('dirpg_cpp', ['dirpg.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
