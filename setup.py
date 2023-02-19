from setuptools import setup, find_packages


setup(
    packages=find_packages(
        exclude=["docs", "plot"]
    ),
    name="irl-gym",
    version="2.0.0",
    install_requires=["numpy", 
                      "gymnasium", 
                      "pygame>=2.1.0"],
)
