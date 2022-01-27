from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    
setup(name='NPST3',
      version='1.0',
      description='Implementation of Neural Policy Style Transfer with Twin-Delayed DDPG (NPST3) framework for Shared Control of Robotic Manipulators',
      url='https://github.com/RaulFdzbis/NPST3',
      author='Raul Fernandez Fernandez',
      author_email='raulfernandezbis@gmail.com',
      packages=find_packages(),
      install_requires = requirements,
 )
