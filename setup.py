from setuptools import setup, find_packages

setup(
    name="SegmentationModels",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "pandas",
        "keras",
        "matplotlib",
        "seaborn"
    ],
)