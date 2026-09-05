"""
Setup script for Privacy-First Federated Learning Platform
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="privacy-first-federated-learning",
    version="2.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Privacy-first federated learning platform with real-time differential privacy controls",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/privacy-first-federated-learning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "fl-server=src.perfect_federated_platform_v3:main",
            "fl-dashboard=src.enhanced_dashboard_v4:main",
            "fl-client=src.perfect_federated_platform_v3:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": [
            "config.json",
            "docker-compose.yml",
            "Dockerfile",
            "data/*",
            "logs/*",
        ],
    },
    keywords=[
        "federated learning",
        "differential privacy",
        "machine learning",
        "privacy-preserving",
        "secure aggregation",
        "ai security",
        "cryptographic ml",
        "distributed learning",
        "privacy by design",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/privacy-first-federated-learning/issues",
        "Source": "https://github.com/yourusername/privacy-first-federated-learning",
        "Documentation": "https://privacy-first-federated-learning.readthedocs.io/",
    },
)
