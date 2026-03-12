from setuptools import setup, find_packages

setup(
    name='marill_trainer',
    version='0.1',
    packages=find_packages(),
    description='MARILL Trainer',
    long_description='A variant of HuggingFace Trainer that does fine-tuning for MARILL',
    author='Anonymous',
    install_requires=[
        'transformers>=4.31.0',
        'accelerate>=0.21.0',
        'flash-attn==2.1.0'
        # any other packages that your package depends on
        # e.g. 'requests', 'numpy', etc.
    ]
)