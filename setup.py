from setuptools import setup, find_packages

with open('./requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="ML-Navigator",
    version="0.0.1",
    python_requires='>=3.6',
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    author="Maher Deeb",
    description="ML-Navigator is a tutorial-based Machine Learning framework. The main component of ML-Navigator is "
                "the flow. A flow is a collection of compact methods/functions that can be stuck together with guidance"
                " texts.",
    keywords="Machine Learning Data Science ML-Navigator tutorial-based data pipelines flows flow",
    url="https://www.ki-labs.com/",
    project_urls={
        "Bug Tracker": "https://github.com/KI-labs/ML-Navigator/issues",
        "Documentation": "https://ki-labs.github.io/ML-Navigator/index.html",
    },
    classifiers=[
        'Programming Language :: Python :: 3.6'
        'License :: OSI Approved :: Apache Software License'
    ]
)
