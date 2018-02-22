# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 11:00:57 2016

@author: courbot
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("image_tools.pyx")
)

setup(
    ext_modules = cythonize("fields_tools.pyx")
)

setup(
    ext_modules = cythonize("gibbs_sampler.pyx")
)
