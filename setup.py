from setuptools import setup

with open('requirements.txt', 'r') as fin:
    reqs = fin.read().split('\n')

setup(
    name='adaptiveleak',
    version='1.0.0',
    author='Tejas Kannan',
    email='tkannan@uchicago.edu',
    description='Removing information leakage from adaptive sampling protocols.',
    packages=['adaptiveleak'],
    install_requires=reqs
)
