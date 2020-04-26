from setuptools import setup, find_packages
NAME = 'dgm1'

VERSION = '0.0.1'

DESCRIPTION = 'DEM from NRW (Germany)'

LONG_DESCRIPTION = 'Processing Digitales Geländemodel (Digital Elevation Model) data from NRW (Germany)'

CLASSIFIERS = [  # https://pypi.python.org/pypi?:action=list_classifiers
    'Development Status :: 4 - Beta',
    'Environment :: Console',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Natural Language :: English',
    'Operating System :: OS Independent',
    'Programming Language :: Python :: 3 :: Only',
    'Topic :: Scientific/Engineering :: GIS'
]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    download_url='https://github.com/JRoehrig/dgm1',
    url='https://github.com/JRoehrig/dgm1',
    author='Jackson Roehrig',
    author_email='Jackson.Roehrig@th-koeln.de',
    license='MIT',
    classifiers=CLASSIFIERS,
    install_requires=['gdal'],
    packages=find_packages(),
    scripts=[]
)

