import os
import sys
from setuptools import setup, find_packages


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
      extras_require={'class': ['cython', 'pyclass @ git+https://github.com/adematti/pyclass'], 'camb': ['camb'], 'isitgr': ['isitgr'], 'astropy': ['astropy'], 'extras': ['pyfftw'], 'jax': ['jax', 'interpax @ git+https://github.com/adematti/interpax']},
      packages=find_packages(),
      package_data={package_basename: ['data/*.dat', 'data/*.csv', 'bindings/*/*.yaml', 'emulators/train/*/emulator.npy']})
