from setuptools import setup

PACKAGE_DIR = ["sparsekmeans"]
PACKAGE_NAME = "sparsekmeans"
VERSION = "0.1"


# license parameters
license_source = ""
license_file = "LICENSE"
license_name = "MIT"


setup(
    name=PACKAGE_NAME,
    packages=PACKAGE_DIR,
    version=VERSION,
    python_requires=">=3.10",
    install_requires=["numpy","python-graphblas","scipy","libsvm-official>=3.36.0"],
    description="",
    long_description_content_type="",
    author="Chih-Jen Lin, He-Zhe Lin, Khoi Nguyen Pham Dang",
    author_email="cjlin@csie.ntu.edu.tw",
    url="https://github.com/cjlin1/sparsekmeans",
    license=license_name,
)