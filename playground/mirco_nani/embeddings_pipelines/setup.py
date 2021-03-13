import setuptools
setuptools.setup(
    name="soundofai-osr-embeddings-pipelines",
    version="0.1.0",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=open("requirements/prod.txt","r").readlines()
)