from distutils.core import setup

setup(
    name="at_ml",
    version="0.1.0",
    author="Mustafa Hasanbulli",
    author_email="mustafa@hasanbul.li",
    packages=["at_ml"],
    description="A package to predicting fraudulent transactions.",
    long_description=open("README.md").read(),
    license=open("LICENSE.md").read(),
    entry_points={
        'console_scripts': ['atml-cli=at_ml.cmd:main']
    }

)