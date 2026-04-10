from setuptools import setup, find_packages

setup(
    name='visionlaw',
    version='1.0',
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        'numpy',
        'torch',
        'warp-lang',
        'pyvista',
        'trimesh',
        'tqdm',
        'tensorboard',
    ],
)
