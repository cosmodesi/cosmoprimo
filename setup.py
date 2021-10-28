from setuptools import setup


setup(name='cosmoprimo',
      version='0.0.1',
      author='Arnaud de Mattia',
      author_email='',
      description='Lightweight primordial cosmology package, including wrappers to CLASS, CAMB, Eisenstein and Hu...',
      license='GPL3',
      url='http://github.com/adematti/cosmoprimo',
      install_requires=['numpy','scipy'],
      extras_require={'extras':['cython','pyclass @ git+https://github.com/adematti/pyclass','camb','pyfftw']},
      package_data={'cosmoprimo': ['data/*.dat']},
      packages=['cosmoprimo']
)
