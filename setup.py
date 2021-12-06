import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="occupiedgpus",
    version="0.0.8",
    author="Zhichao Jin",
    author_email="jinzcdev@icloud.com",
    description="The program for occupation of GPUs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jinzcdev/occupied-gpus.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['torch', 'nvidia-ml-py'],
)

'''
python3 -m pip install --user --upgrade setuptools wheel
python3 setup.py sdist bdist_wheel
python3 -m twine upload dist/* -u jinzcdev
'''
