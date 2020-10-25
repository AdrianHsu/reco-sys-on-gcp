import setuptools

# NOTE: Any additional file besides the `main.py` file has to be in a module
#       (inside a directory) so it can be packaged and staged correctly for
#       cloud runs.

REQUIRED_PACKAGES = [
    'apache-beam[gcp]==2.17.0',
    'tensorflow-transform==0.21.2',
    'tensorflow==1.15.0',
]

setuptools.setup(
    name='ahsu-movielens',
    version='0.0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages(),
    include_package_data=True,
    description='Cloud ML movielens sample with preprocessing @ahsu',
)
