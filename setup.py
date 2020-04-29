import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='householdenv',
    version='0.0.4',
    packages=setuptools.find_packages(),
    install_requires=["gym", "numpy"],  # And any other dependencies foo needs
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*.txt", "*.json"]
    },
    # metadata to display on PyPI
    author="Diego Cabo Golvano",
    author_email="dcgdiego@gmail.com",
    description="This is an environment for RL purposes",  # TODO
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="Reinforcement Learning environment",
    url="https://github.com/mrcabo/Household-env.git",  # project home page, if any
    project_urls={
        "Source Code": "https://github.com/mrcabo/Household-env.git",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
