from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='dirpg_cpp',
      ext_modules=[cpp_extension.CppExtension('dirpg_cpp',
                                              sources=['dirpg.cpp',
                                                       'retsp/batched_graphs.cpp',
                                                       'retsp/a_star_sampling.cpp',
                                                       'retsp/batched_heaps.cpp',
                                                       'retsp/batched_trajectories.cpp',
                                                       'retsp/node_allocator.cpp',
                                                       'retsp/mst_node.cpp',
                                                       'retsp/info_node.cpp',
                                                       'retsp/gumbel_state.cpp',
                                                       'retsp/union_find.cpp',
                                                      ],
                                              include_dirs=['retsp'])],
      headers=[
               'retsp/batched_graphs.h',
               'retsp/a_star_sampling.h',
               'retsp/batched_heaps.h',
               'retsp/batched_trajectories.h',
               'retsp/node_allocator.h',
               'retsp/mst_node.h',
               'retsp/info_node.h',
               'retsp/gumbel_state.h',
               'retsp/union_find.h'],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

