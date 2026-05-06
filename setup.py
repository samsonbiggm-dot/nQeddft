# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name="nqeddft",
    version="0.1.0",
    description="QED-DFT software built on PySCF — photon-electron and photon-phonon coupling",
    author="Your Group",
    python_requires=">=3.9",
    packages=["nqeddft"] + ["nqeddft." + p for p in find_packages()],
    package_dir={"nqeddft":"."},
    install_requires=[
        "pyscf>=2.3",
        "numpy>=1.21",
        "scipy>=1.7",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "pytest-cov"],
        "gpu": ["gpu4pyscf>=0.6"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
