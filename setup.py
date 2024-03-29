import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="sklearned",
    version="0.0.2",
    description="Surrogates",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/microprediction/sklearned",
    author="microprediction",
    author_email="peter.cotton@microprediction.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["sklearned"],
    test_suite='pytest',
    tests_require=['pytest'],
    include_package_data=True,
    install_requires=["wheel","pathlib","keras","keras-tcn"],
    entry_points={
        "console_scripts": [
            "sklearned=sklearned.__main__:main",
        ]
    },
)
