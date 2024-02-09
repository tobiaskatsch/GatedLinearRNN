from setuptools import setup, find_packages

setup(
    name='flax-gate-loop',
    version='1.0.0',
    author='Tobias Katsch',
    author_email='tobias.katsch42@gmail.com',
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        'jax>=0.4.20',
        'flax>=0.8.0',
    ],
    url='https://github.com/tobiaskatsch/GateLoop',
    license='Apache License, Version 2.0',
    description='GateLoop Model',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
)
