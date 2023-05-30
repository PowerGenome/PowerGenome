from setuptools import find_packages, setup

setup(
    name="powergenome",
    packages=find_packages(),
    version="0.6.0",
    description="Extract PUDL data for use in power system models",
    author="Greg Schivley",
    entry_points={
        "console_scripts": [
            "run_powergenome_multiple = powergenome.run_powergenome_multiple_outputs_cli:main",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.9,<3.12",
    install_requires=[
        "catalystcoop.pudl>=0.6.0, <=2022.11.30",
        "beautifulsoup4",
        "fastparquet",
        "statsmodels",
        "python-dotenv",
        "flatten-dict",
        "ruamel.yaml",
        "pyyaml",
        "frozendict",
        "openpyxl>=3.0",
        "geopandas",
        "descartes",
    ],
    extras_require={
        "dev": [
            "black>=23",  # A deterministic code formatter
            "isort>=5,<6",  # Standardized import sorting
            "tox>=3.20,<5",  # Python test environment manager
            "twine>=3.3,<5.0",  # Used to make releases to PyPI
        ],
        "tests": [
            "bandit>=1.6,<2",  # Checks code for security issues
            "coverage>=5.3,<8",  # Lets us track what code is being tested
            "doc8>=0.9,<1.1",  # Ensures clean documentation formatting
            "flake8>=4,<7",  # A framework for linting & static analysis
            "flake8-builtins>=1.5,<3",  # Avoid shadowing Python built-in names
            "flake8-colors>=0.1.9,<0.2",  # Produce colorful error / warning output
            "flake8-docstrings>=1.5,<2",  # Ensure docstrings are formatted well
            "flake8-rst-docstrings>=0.2,<0.4",  # Allow use of ReST in docstrings
            "flake8-use-fstring>=1,<2",  # Highlight use of old-style string formatting
            "pytest>=6.2,<8",  # Our testing framework
            "pytest-cov>=2.10,<5.0",  # Pytest plugin for working with coverage
        ],
    },
)
