from distutils.core import setup, Extension
import os


setup(name='flipperPol',
      version='0.1',
      description='Polarization map manipulation tools',
      author='Thibaut Louis',
      packages=['flipperPol'],
      install_requires=['healpy>=1.10.3',
                        'numpy>=1.11.3',
                        'astropy>=1.3',
                        'matplotlib>=1.5.3',
                        'pyfits',
                        'scipy>=0.18.1',
                        'astLib>=0.8.0'],
      zip_safe=False)
