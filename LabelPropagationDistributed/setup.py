import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "LabelPropagationDistributed",
    author = "Madhura Gadgil",
    author_email = "mgadgil09@gmail.com",
    description = ("A semi-supervised approach for clustering "
                                   "using Label Propagation."),
    #url = "http://packages.python.org/an_example_pypi_project",
    packages=['LabelPropagationDistributed', 'tests'],
    long_description=read('README'),
    
)
