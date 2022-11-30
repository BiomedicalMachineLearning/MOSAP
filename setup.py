"""Install package."""
import io
import re
from setuptools import setup, find_packages
from pkg_resources import parse_requirements

def read_version(filepath: str) -> str:
    """Read the __version__ variable from the file.
    Args:
        filepath: probably the path to the root __init__.py
    Returns:
        the version
    """
    match = re.search(
        r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        io.open(filepath, encoding="utf_8_sig").read(),
    )
    if match is None:
        raise SystemExit("Version number not found.")
    return match.group(1)


# with open("requirements.txt") as f:
#     requirements = f.read().splitlines()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

with open("README.md", encoding="utf8") as readme_file:
    readme = readme_file.read()

# ease installation during development
vcs = re.compile(r"(git|svn|hg|bzr)\+")
try:
    with open("requirements.txt") as fp:
        VCS_REQUIREMENTS = [
            str(requirement)
            for requirement in parse_requirements(fp)
            if vcs.search(str(requirement))
        ]
except FileNotFoundError:
    # requires verbose flags to show
    print("requirements.txt not found.")
    VCS_REQUIREMENTS = []

setup_requirements = []
setup(
    author="Genomics and Machine Learning lab",
    author_email="uqmtra12@uq.edu.au",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description="A Multi-Omics Spatial Analysis Platform (MOSAP)",
    install_requires=VCS_REQUIREMENTS,
    license="BSD license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="MOSAP",
    name="MOSAP",
    packages=find_packages(),
    setup_requires=setup_requirements,
    url="https://github.com/BiomedicalMachineLearning/MOSAP",
    version=read_version("mosap/__init__.py"),
    zip_safe=False,
)
