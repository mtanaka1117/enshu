from setuptools import setup, find_packages

with open('requirements.txt', 'r') as fin:
    reqs = fin.read().split('\n')

setup(
    name='adaptiveleak',
    version='1.0.0',
    description='Removing information leakage from adaptive sampling protocols.',
    packages=find_packages(),
    install_requires=reqs
)
