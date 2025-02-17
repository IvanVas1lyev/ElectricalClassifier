from setuptools import setup

setup(
    name="my_classifier",
    version="0.1",
    py_modules=["my_classifier"],
    install_requires=[
        "numpy",
        "scikit-learn",
    ],
)