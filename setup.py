from setuptools import setup, find_packages

setup(
    name='faster-rcnn',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'Pillow',
        # Add other dependencies here
    ],
    entry_points={
        'console_scripts': [
            'faster-rcnn=main:main',
        ],
    },
)