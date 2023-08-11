from pathlib import Path
import os
from setuptools import find_packages, setup

current_dir = Path(__file__).parent
long_description = (current_dir / "README.md").read_text()


def _get_requirements():
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
    return requirements


def _get_version():
    if os.environ.get('CI_COMMIT_TAG', None):
        version = os.environ['CI_COMMIT_TAG']
    else:
        version = _get_saved_version()
    version = version.replace("v", "")
    version_split = version.split(".")
    assert len(version_split) == 3, f"Unknown version format {version}, expecting vX.Y.Z"
    return version


def _get_saved_version():
    # Added in during CI/CD so that correct version is used when user
    # installs.
    try:
        with open("./version.txt") as f:
            version = f.read().rstrip()
    except Exception:
        version = "v0.0.1"
    return version


if __name__ == "__main__":
    setup(
        name='chai-guanaco',
        version=_get_version(),
        description='Chai Guanaco',
        author='Chai Research Corp.',
        author_email='hello@chai-research.com',
        license='MIT',
        packages=['chai_guanaco'],
        package_dir={"": "src"},
        package_data={"chai_guanaco": ["resources/*"]},
        url='https://www.chai-research.com',
        zip_safe=False,
        long_description=long_description,
        long_description_content_type='text/markdown',
        install_requires=_get_requirements(),
        include_package_data=True,
        entry_points={
            'console_scripts': ['chai-guanaco=chai_guanaco.login_cli:cli'],
        },
    )
