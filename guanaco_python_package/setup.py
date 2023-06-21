from pathlib import Path
import os
from setuptools import find_packages, setup

current_dir = Path(__file__).parent
long_description = (current_dir / "README.md").read_text()

if os.environ.get('CI_COMMIT_TAG', None):
    version = os.environ['CI_COMMIT_TAG']
else:
    version = "0.0.3"

setup(
    name='chai-guanaco',
    version=version,
    description='Chai Guanaco',
    author='Chai Research Corp.',
    author_email='hello@chai-research.com',
    license='MIT',
    packages=["chai_guanaco"],
    package_dir={"chai_guanaco": "./src/chai_guanaco/"},
    url='https://www.chai-research.com',
    zip_safe=False,
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=['requests']
)
