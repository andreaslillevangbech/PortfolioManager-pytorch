from setuptools import setup, find_packages

with open("README.md") as f:
    readme = f.read()

setup(
    name="PortfolioManager",
    version="1.0.0",
    description="",
    long_description=readme,
    author="",
    author_email="",
    packages=find_packages(exclude=("tests", "docs"),
                           include=("matplotlib", "tensorflow", "xarray",
                                    "pandas", "scipy")))
