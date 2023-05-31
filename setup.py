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
)
