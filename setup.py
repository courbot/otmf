from setuptools import setup

setup(
    name='otmf',
    version='2.0',
    packages=['otmf'],
    package_dir={'otmf': 'lib'},
    zip_safe=False,
    install_requires=['numpy', 'scipy'],
)
