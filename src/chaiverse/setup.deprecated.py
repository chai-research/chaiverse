import os
from pathlib import Path
from setuptools import find_packages, setup

long_description = '''
Chai Guanaco](https://www.chai-research.com/competition.html) is part of the Chai Prize Competition, accelerating community AGI. It's the world's first open community challenge with real-user evaluations. You models will be directly deployed on the [Chai App](http://tosto.re/chaiapp) where our over 500K daily active users will be providing live feedback. Get to top of the leaderboard and share the $1 million cash prize!
'''

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
        package_dir={
            "chai_guanaco": "."
        },
        package_data={
            "chai_guanaco": ["resources/*"],
        },
        url='https://www.chai-research.com',
        zip_safe=False,
        long_description=long_description,
        long_description_content_type='text/markdown',
        install_requires=_get_requirements(),
        include_package_data=True,
        entry_points={
            'console_scripts': [
                'chai-guanaco=login_cli:cli'
            ],
        },
    )
