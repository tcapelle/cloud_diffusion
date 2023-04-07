from setuptools import setup, find_packages

exec(open("cloud_diffusion/version.py").read())

setup(
    name="cloud_diffusion",
    packages=find_packages(),
    version=__version__,
    license="MIT",
    description="Diffusion on the Clouds: Short-term solar energy forecasting with Diffusion Models",
    author="Thomas Capelle",
    author_email="tcapelle@pm.me",
    url="https://github.com/tcapelle/cloud_diffusion",
    long_description_content_type="text/markdown",
    keywords=[
        "artificial intelligence",
        "generative models",
        "pytorch",
        "stable diffusion",
    ],
    install_requires=[
        "torch",
        "fastprogress",
        "fastcore",
        "wandb",
        "numpy",
        "matplotlib",
        "diffusers",
        "denoising_diffusion_pytorch",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
)
