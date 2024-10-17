import re
import setuptools
from pkg_resources import get_distribution, DistributionNotFound

with open("README.md", "r") as fh:
    long_description = fh.read()

INSTALL_REQUIRES = [
    "accelerate==0.34.2",
    "beautifulsoup4==4.12.3",
    "datasets==3.0.1",
    "ftfy==6.3.0",
    "numpy==1.26.4",
    "spacy==3.7.5",
    "tiktoken==0.7.0",
    "torch==2.1.0",
    "torchinfo==1.8.0",
    "torchvision==0.16.0",
    "tqdm==4.66.5",
]

DEPENDENCY_LINKS = [
    "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl",
]

ALT_INSTALL_REQUIRES = {}


def check_alternative_installation(install_require, alternative_install_requires):
    """If some version version of alternative requirement installed, return alternative,
    else return main.
    """
    for alternative_install_require in alternative_install_requires:
        try:
            alternative_pkg_name = re.split(r"[!<>=]", alternative_install_require)[0]
            get_distribution(alternative_pkg_name)
            return str(alternative_install_require)
        except DistributionNotFound:
            continue

    return str(install_require)


def get_install_requirements(main_requires, alternative_requires):
    """Iterates over all install requires
    If an install require has an alternative option, check if this option is installed
    If that is the case, replace the install require by the alternative to not install dual package
    """
    install_requires = []
    for main_require in main_requires:
        if main_require in alternative_requires:
            main_require = check_alternative_installation(
                main_require, alternative_requires.get(main_require)
            )
        install_requires.append(main_require)

    return install_requires


INSTALL_REQUIRES = get_install_requirements(INSTALL_REQUIRES, ALT_INSTALL_REQUIRES)

setuptools.setup(
    name="minllm",
    version="0.0.1",
    author="Sam Wookey",
    author_email="sam@thinky.ai",
    description="Minimum Language Model Implementations",
    long_description_content_type="text/markdown",
    long_description=long_description,
    license_files=("LICENSE",),
    license="Copyright 2024, Sam Wookey",
    url="https://github.com/swookey-thinky/minllm",
    packages=setuptools.find_packages(
        exclude=["config", "docs", "sampling", "tools", "training"]
    ),
    include_package_data=True,
    package_data={"": ["*.json", "*.bpe", "*.csv"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    install_requires=INSTALL_REQUIRES,
    dependency_links=DEPENDENCY_LINKS,
    python_requires=">=3.6",
)
