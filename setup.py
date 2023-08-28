from pathlib import Path
from setuptools import find_packages, setup

current_dir = Path(__file__).parent
long_description = (current_dir / "README.md").read_text()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="chaiverse",
    url='https://www.chai-research.com',
    author='Chai Research Corp.',
    author_email='hello@chai-research.com',
    version="0.1",
    description="Chaiverse",
    install_requires=requirements,
    include_package_data=True,
    long_description=long_description,
    license='MIT',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    long_description_content_type='text/markdown',
)

