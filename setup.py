"""Setup script for the Universal ML/AI Research & Learning Platform"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='dgx-ml-platform',
    version='1.0.0',
    description='Universal ML/AI Research & Learning Platform',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='ML Research Team',
    author_email='ml.research@example.com',
    url='https://github.com/yourusername/dgx_code',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='machine-learning deep-learning ai computer-vision nlp robotics',
    project_urls={
        'Documentation': 'https://dgx-ml-platform.readthedocs.io',
        'Source': 'https://github.com/yourusername/dgx_code',
        'Tracker': 'https://github.com/yourusername/dgx_code/issues',
    },
    entry_points={
        'console_scripts': [
            'train=framework.cli:train',
            'evaluate=framework.cli:evaluate',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
