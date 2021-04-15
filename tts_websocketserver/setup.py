import setuptools
setuptools.setup(
    name="soundofai-osr-tts-websocketserver",
    version="0.1.0",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=open("requirements.txt","r").readlines()
)