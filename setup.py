#!/usr/bin/env python
import os
from setuptools import setup

setup(
    name = "ASHLE",
    version = "1.0.0",
    author = "Iran R. Roman",
    author_email = "roman@nyu.edu",
    description = ("Adaptive Synchronization with Hebbian Learning and Elasticity"),
    license = "BSD",
    keywords = "Adaptive Synchronization, Hebbian Learning, Nonlinear dynamics, Oscillators, Synchronization",
    url = "https://github.com/iranroman/ASHLE.git",
    packages = ['exp1', 'exp2', 'exp3'],
    install_requires = [
        'matplotlib>=3.5.3',
        'numpy>=1.21.6',
        'scipy>=1.7.3'
    ]
)
