from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="DeepSecE",
    version="0.1.2",
    description="A Deep Learning Framework for Multi-class Secreted Protein Prediction in Gram-negative Bacteria.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Yumeng Zhang",
    url="https://github.com/zhangyumeng1sjtu/DeepSecE",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        'torch',
        'biopython',
        'einops',
        'fair-esm',
        'tqdm',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'tensorboardX',
        'umap-learn',
        'warmup-scheduler',
    ]
)
