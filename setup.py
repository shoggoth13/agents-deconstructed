from setuptools import setup, find_packages

setup(
    name="agents_deconstructed",
    version="0.0.2",
    description="Utils for running 'agents' where the core logic is not obfuscated.",
    packages=find_packages(),
    install_requires=[
        "langchain",
    ],
)
