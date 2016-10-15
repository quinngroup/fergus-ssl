try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'name' : "LabelPropagationDistributed",
    'author' : "Madhura Gadgil",
    'author_email' : "mgadgil09@gmail.com",
    'description'  "A semi-supervised approach for clustering "
                                   "using Label Propagation.",
    'url' : "https://github.com/quinngroup/fergus-ssl.git",
    'version' : "0.1",
    'install_requires': ['pytest'],
    'packages':['LabelPropagationDistributed'],
    }

setup(**config)
