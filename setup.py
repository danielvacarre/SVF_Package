from pathlib import Path
from setuptools import setup, find_packages

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

VERSION = '0.0.2'
DESCRIPTION = 'Paquete Support Vector Frontier'
PACKAGE_NAME = 'svf_package'
AUTHOR = 'Daniel Valero Carreras'
EMAIL = 'dvalero@umh.es'
GITHUB_URL = 'https://github.com/danielvacarre'

setup(
    name = PACKAGE_NAME,
    packages = find_packages(),
    version = VERSION,
    license='MIT',
    description = DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    author = AUTHOR,
    author_email = EMAIL,
    url = GITHUB_URL,
    keywords = [],
    install_requires=[
        "docplex", "joblib","numpy","pandas","python-dateutil","pytz",
        "scikit-learn", "scipy", "six", "threadpoolctl"
    ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)