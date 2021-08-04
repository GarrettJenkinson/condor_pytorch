# Sebastian Raschka 2020
# condor_pytorch
# Author: Garrett Jenkinson <Jenkinson.William@mayo.edu>
#
# License: MIT

from os.path import realpath, dirname, join
from setuptools import setup, find_packages
import condor_pytorch

VERSION = condor_pytorch.__version__
PROJECT_ROOT = dirname(realpath(__file__))

REQUIREMENTS_FILE = join(PROJECT_ROOT, 'requirements.txt')

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()

install_reqs.append('setuptools')

setup(name='condor_pytorch',
      version=VERSION,
      description='CONDOR ordinal regression for PyTorch',
      author='Garrett Jenkinson',
      author_email='Jenkinson.William@mayo.edu',
      url='https://github.com/GarrettJenkinson/condor_pytorch',
      packages=find_packages(),
      package_data={'': ['LICENSE.txt',
                         'README.md',
                         'requirements.txt']
                    },
      include_package_data=True,
      setup_requires=[],
      install_requires=install_reqs,
      extras_require={'testing': ['pytest'],
                      'docs': ['mkdocs']},
      license='MIT',
      platforms='any',
      keywords=['deep learning', 'pytorch', 'AI'],
      classifiers=[
             'License :: OSI Approved :: MIT License',
             'Development Status :: 5 - Production/Stable',
             'Operating System :: Microsoft :: Windows',
             'Operating System :: POSIX',
             'Operating System :: Unix',
             'Operating System :: MacOS',
             'Programming Language :: Python :: 3.7',
             'Topic :: Scientific/Engineering',
             'Topic :: Scientific/Engineering :: Artificial Intelligence',
             'Topic :: Scientific/Engineering :: Information Analysis',
             'Topic :: Scientific/Engineering :: Image Recognition',
      ],
      long_description_content_type='text/markdown',
      long_description="""

Library implementing the core utilities for the
CONDOR ordinal regression approach from

TBD publication.
""")
