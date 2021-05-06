from setuptools import setup

with open('requirements.txt', 'r') as fh:
    install_requires = [l.strip() for l in fh]

extras_require = {}
with open('requirements-extras.txt', 'r') as fh:
    extras_require['extras'] = [l.strip() for l in fh]
    extras_require['full'] = extras_require['extras']

print(extras_require)

setup(name='cosmoprimo',
      version='0.0.1',
      author='Arnaud de Mattia',
      author_email='',
      description='Lightweight primordial cosmology package, including wrappers to CLASS, CAMB, Eisenstein and Hu...',
      license='GPL3',
      url='http://github.com/adematti/cosmoprimo',
      install_requires=install_requires,
      extras_require=extras_require,
      packages=['cosmoprimo']
)