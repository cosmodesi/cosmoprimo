import os
import sys
from setuptools import setup


package_basename = 'cosmoprimo'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), package_basename))
import _version
version = _version.__version__


setup(name=package_basename,
      version=version,
      author='cosmodesi',
      author_email='',
      description='Lightweight primordial cosmology package, including wrappers for CLASS, CAMB, Eisenstein and Hu...',
      license='BSD3',
      url='http://github.com/cosmodesi/cosmoprimo',
      install_requires=['numpy', 'scipy'],
      extras_require={'class': ['cython', 'pyclass @ git+https://github.com/adematti/pyclass@1.0.0'], 'camb': ['camb'], 'astropy': ['astropy'], 'extras': ['pyfftw']},
      package_data={'cosmoprimo': ['data/*.dat', 'data/*.csv']},
      packages=[package_basename])
