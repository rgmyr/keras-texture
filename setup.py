import os
from setuptools import setup, find_packages

requirements = [
    'numpy',
    'scikit-image',
    'keras>=2.0.0',
    'matplotlib',
]
# TODO: enforce tensorflow backend

setup(name='keras-texture',
      version='0.1',
      author='Ross Meyer',
      author_email='ross.meyer@utexas.edu',
      description='Keras Texture Package.',
      url='https://github.com/rgmyr/keras-texture',
      packages=find_packages(),
      install_requires=requirements,
      zip_safe=False
)
