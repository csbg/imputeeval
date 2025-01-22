from setuptools import setup, find_packages

setup(
    name="imputeeval",  # Package name
    version="0.1.0",  # Initial version
    description="A framework for evaluating imputation methods with artificial NA values.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Thomas Rauter",
    author_email="thomas.rauter@plus.ac.at",
    url="https://github.com/Thomas-Rauter/imputeeval",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.0",
        "numpy>=2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0",
        ],
    },
    python_requires=">=3.10",  # Minimum Python version
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
