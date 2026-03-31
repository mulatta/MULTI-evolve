from setuptools import setup, find_packages

setup(
    name="multievolve",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    scripts=[
        "scripts/p1_train.py",
        "scripts/p2_propose.py",
        "scripts/p3_assembly_design.py",
        "scripts/plm_zeroshot_ensemble.py",
    ],
    install_requires=[],
    python_requires=">=3.11",
    authors="Vincent Q. Tran, Matthew Nemeth, and Brian Hie",
    description="MULTI-evolve: model-guided, universal, targeted installation of multi-mutants",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/VincentQTran/multievolve",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
