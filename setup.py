from setuptools import setup, find_packages


def _load_requires():
    return open("requirements.txt").read().splitlines()


setup(
    name="nenepy-summary",
    version="0.0.1",
    install_requires=_load_requires(),
    packages=find_packages(),
    author="Nenetti",
    url="https://github.com/Nenetti/nenepy-summary",
)
