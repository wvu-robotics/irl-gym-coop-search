from setuptools import setup, find_packages


setup(
    packages=find_packages(
        exclude=["docs", "plot"]
    ),
    name="irl_gym",
    version="2.0.0",
    install_requires=[],
)
