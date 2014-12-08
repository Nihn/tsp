from setuptools import setup

setup(
    author='Mateusz Moneta',
    author_email='mateuszmoneta@gmail.com',
    name='tsp',
    install_requires=[
        'numpy',
        'matplotlib',
        'networkx',
        'progressbar2==2.7.3',
        'argh==0.26.1',
    ],
    version='1.0.0',
    packages=['tsp'],
)
