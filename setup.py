#!/usr/bin/env python3
"""
Setup script for Video Penibility Assessment Framework.

This package provides a comprehensive deep learning framework for assessing
physical penibility (strain/difficulty) in videos using multiple feature
extraction methods and temporal modeling approaches.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    requirements = requirements_file.read_text().strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="video-penibility-assessment",
    version="1.0.0",
    author="Salim Khazem",
    author_email="salimkhazem97@gmail.com",
    description="Deep learning framework for video penibility assessment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/salimkhazem/video-penibility",
    project_urls={
        "Bug Reports": "https://github.com/salimkhazem/video-penibility/issues",
        "Source": "https://github.com/salimkhazem/video-penibility",
        "Documentation": "https://github.com/salimkhazem/video-penibility#readme",
    },
    
    # Package discovery
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=requirements,
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-xdist>=3.0.0",
            "pytest-mock>=3.10.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
            "bandit>=1.7.0",
            "safety>=2.3.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "sphinx-autodoc-typehints>=1.19.0",
        ],
        "profiling": [
            "memory-profiler>=0.60.0",
            "line-profiler>=4.0.0",
            "psutil>=5.9.0",
        ],
    },
    
    # Entry points for CLI commands
    entry_points={
        "console_scripts": [
            "train-penibility=video_penibility.cli.train:main",
            "evaluate-penibility=video_penibility.cli.evaluate:main",
        ],
    },
    
    # Package metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    
    # Keywords for PyPI search
    keywords=[
        "machine-learning",
        "deep-learning",
        "computer-vision", 
        "video-analysis",
        "penibility-assessment",
        "temporal-modeling",
        "pytorch",
        "transformers",
        "i3d",
        "swin3d",
    ],
    
    # Include additional files
    include_package_data=True,
    package_data={
        "video_penibility": [
            "configs/*.yaml",
            "*.yaml",
        ],
    },
    
    # Zip safety
    zip_safe=False,
) 