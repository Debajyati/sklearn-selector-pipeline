from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Read version
exec(open("sklearn_selector_pipeline/_version.py").read())

setup(
    name="sklearn-selector-pipeline",
    version=__version__,
    author="Debajyati",
    author_email="ddebajyati@gmail.com",
    description="Meta-estimators that combine feature selectors with classifiers and regressors for seamless sklearn integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Debajyati/sklearn-selector-pipeline",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "scikit-learn>=1.0.0",
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
            "pre-commit",
            "build>=1.2.2.post1",
            "twine>=6.1.0",
        ],
        "examples": [
            "matplotlib>=3.0.0",
            "seaborn>=0.11.0",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
            "numpydoc",
        ],
    },
    keywords="scikit-learn feature-selection classification regression machine-learning sklearn pipeline",
    project_urls={
        "Bug Reports": "https://github.com/Debajyati/sklearn-selector-pipeline/issues",
        "Source": "https://github.com/Debajyati/sklearn-selector-pipeline",
        "Documentation": "https://sklearn-selector-pipeline.readthedocs.io/",
    },
)
