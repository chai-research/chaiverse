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
        name='chaiverse',
        version=_get_version(),
        description='Chaiverse',
        author='Chai Research Corp.',
        author_email='hello@chai-research.com',
        license='MIT',
        packages=['chaiverse'],
        package_dir={
            "chaiverse": "."
        },
        package_data={
            "chaiverse": ["resources/bot_config/*"]
        },
        url='https://www.chaiverse.com',
        zip_safe=False,
        long_description=long_description,
        long_description_content_type='text/markdown',
        install_requires=_get_requirements(),
        entry_points={
            'console_scripts': [
                'chaiverse=chaiverse.login_cli:cli',
            ],
        },
    )
