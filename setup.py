from setuptools import find_packages, setup

setup(
    name="powergenome",
    packages=find_packages(),
    version="0.1.0",
    description="Extract PUDL data for use in power system models",
    author="Greg Schivley",
    entry_points={
        "console_scripts": ["powergenome_data = powergenome.extract_pudl_data_cli:main"]
    },
)
