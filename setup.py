"""
lambdata - a collection of Data Science helper functions
"""

import setuptools

REQUIRED = [
    "numpy",
    "pandas",
    "random",
    "unittest"
]

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

setuptools.setup(
    name="lambdata-echiyembekeza",
    version="0.0.1",
    author="echiyembekeza",
    description="A collection of Data Science helper functions",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/echiyembekeza/lambdata-echiyembekeza",
    packages=setuptools.find_packages(),
    python_requires=">=3.5",
    install_requires=REQUIRED,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
