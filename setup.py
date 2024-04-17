from setuptools import setup, find_packages
from typing import List

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

__version__ = '0.0.0'

REPO_NAME = 'Burn_Area_Mapping'
AUTHOR_USER_NAME = 'Geethen'
SRC_REPO = ''
AUTHOR_EMAIL = 'geethen.singh@gmail.com'

def get_requirements(file_path:str) ->List[str]:
    """
    Get a list of required packages based on the requirements.txt file

    Args:
        file_path (str): Path to the requirements.txt file which contains the packages and potentially the
          package versions to be installed
        """
    HYPEN_E_DOT = '-e .'
    requirements = []
    # Read requirements.txt file
    with open(file_path, 'r') as file_obj:
        requirements = file_obj.readlines()
        # Remove /n from list
        requirements = [req.replace("\n", "") for req in requirements]

        # If '-e .' is present in list, remove it
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements


setup(
    name='Burn_Area_Mapping',
    version= __version__,
    packages=find_packages(where = "src"),
    install_requires= get_requirements('requirements.txt'),
    author = AUTHOR_USER_NAME,
    author_email = AUTHOR_EMAIL,
    description = 'System for operational burn area mapping',
    long_description= long_description,
    long_description_content = 'text/markdown',
    url = f"http://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls = {'Bug_Tracker': f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues"},
    package_dir={"": "src"}
    )