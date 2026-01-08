from setuptools import setup,find_packages
from typing import List


def get_requirements() -> List[str]:
    requirements_lst:List[str] = []
    try:
        with open('requirements.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                if requirement and requirement != '-e .':
                    requirements_lst.append(requirement)
                
    except FileNotFoundError:
        print("requirements.txt file NOT found")
    
    return requirements_lst

setup(
    name='CardioVascularDisease',
    version='0.0.1',
    author='gokul20',
    author_email='chinnigokul43@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements(),
)