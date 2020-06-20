# `python setup.py build_ext --inplace`
import numpy as np
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(['perfecthash/_graph.pyx'], annotate=True),
    include_dirs=[np.get_include()],
)
