from setuptools import find_packages, setup

setup(
    name="powergenome",
    packages=find_packages(),
    version="0.4.1",
    description="Extract PUDL data for use in power system models",
    author="Greg Schivley",
    entry_points={
        "console_scripts": [
            "run_powergenome = powergenome.run_powergenome_cli:main",
            "run_powergenome_multiple = powergenome.run_powergenome_multiple_outputs_cli:main",
        ]
    },
)
