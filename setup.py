from setuptools import setup, find_packages
import os

# Read README
def read_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, encoding='utf-8') as f:
            return f.read()
    return ''

# Define dependencies
# Note: torch-geometric dependencies require special installation steps
# Users should install PyTorch and torch-geometric dependencies first:
# pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
# --find-links https://data.pyg.org/whl/torch-2.1.0+cu121.html
install_requires = [
    'torch-scatter>=2.1.0',
    'torch-sparse>=0.6.0',
    'torch-cluster>=1.6.0',
    'torch-spline-conv>=1.2.0',
    'torch-geometric>=2.4.0',
]

setup(
    name='tabulartextmultimodalfusion',
    version='0.1.5',
    author='Nadav Cohen',
    author_email='nadav22799@gmail.com',
    description='A framework for multimodal fusion of tabular and text data.',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/nadav22799/TabularTextMultimodalFusion',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=install_requires,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)