# -*- coding: utf-8 -*-

from setuptools import setup

__author__ = ['Lorenzo Porrini']
__email__ = ['lorenzo.porrini2@studio.unibo.it']

PACKAGE_NAME = 'dp-VAE'
PACKAGE_VERSION = '0.0.1'
DESCRIPTION = 'Distance Preserving Variational AutoEncoder applied to spatial transcriptomics data.'
AUTHOR = 'Lorenzo Porrini'
EMAIL = 'lorenzo.porrini2@studio.unibo.it'
REQUIRES_PYTHON = '>=3'
URL = 'https://github.com/zandoff/VAE_scRNA'
DOWNLOAD_URL = URL

setup(
  name=PACKAGE_NAME,
  version=PACKAGE_VERSION,
  description=DESCRIPTION,
  author=AUTHOR,
  author_email=EMAIL,
  python_requires=REQUIRES_PYTHON,
  install_requires=[],
  url=URL,
  download_url=DOWNLOAD_URL,
  setup_requires=[],
  packages=[
    PACKAGE_NAME,
  ],
  package_data={
    PACKAGE_NAME: [],
  },
  include_package_data=True,
  platforms='any',
  classifiers=[
    'Natural Language :: English',
    'Operating System :: MacOS :: MacOS X',
    'Operating System :: POSIX',
    'Operating System :: POSIX :: Linux',
    'Operating System :: Microsoft :: Windows',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: Implementation :: PyPy'
  ],
  entry_points={
        'console_scripts': [
            'dp-VAE = dp-VAE.main',
        ],
    },
)