from setuptoools import setup, find_packages
from typing import List
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
    name='your_package_name',
    version='0.0.1',
    packages=find_packages(),
    install_requires= get_requirements('requirements.txt'),
    author = 'Geethen',
    author_email = 'geethen.singh@gmail.com'
    )