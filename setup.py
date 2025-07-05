from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='tabulartextmultimodalfusion',
    version='0.1.0',
    author='Your Name',  # Placeholder
    author_email='your.email@example.com',  # Placeholder
    description='A framework for multimodal fusion of tabular and text data.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/your-username/TabularTextMultimodalFusion',  # Placeholder
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=required,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)