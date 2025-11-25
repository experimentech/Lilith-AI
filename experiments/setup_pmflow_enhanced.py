#!/usr/bin/env python3
"""Setup for PMFlow BNN Enhanced v0.3.0 - Lilith Edition."""

from setuptools import setup, find_packages
from pathlib import Path

# Read version
version_file = Path(__file__).parent / "pmflow_bnn_enhanced" / "version.py"
version_dict = {}
with open(version_file) as f:
    exec(f.read(), version_dict)

__version__ = version_dict["__version__"]

# Read long description
readme_file = Path(__file__).parent / "pmflow_bnn_enhanced" / "ENHANCEMENTS_v0.3.0.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="pmflow-bnn-enhanced",
    version=__version__,
    description="PMFlow BNN Enhanced for Lilith Neuro-Symbolic AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Lilith AI Project",
    author_email="",
    url="https://github.com/experimentech/Pushing-Medium",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
)
