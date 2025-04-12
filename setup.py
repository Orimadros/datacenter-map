from setuptools import setup, find_packages

setup(
    name="dc_map_project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "geopandas",
        "shapely",
        "jupyter",
        "notebook",
    ],
    author="Leonardo Gomes",
    author_email="leocg@outlook.com",
    description="A project for analyzing and visualizing data center and infrastructure distribution",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/dc_map_project",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
) 