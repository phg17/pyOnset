"""
Setup file
"""
import setuptools

VERS = {}
with open("./pyonset/version.py") as fp:
    exec(fp.read(), VERS)

setuptools.setup(
    name='pyonset',
    version=VERS['__version__'],
    packages=setuptools.find_packages(),
    license='MIT',
    author='Pierre Guilleminot',
    description='Extracting vowels onsets from audio'
)