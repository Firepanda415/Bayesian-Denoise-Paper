#!/usr/bin/env python
# Author: Muqing Zheng


# Go to parent directory, run python setup.py sdist
# Then
# pip install -e /path/to/package


from distutils.core import setup

setup(name='BQER',
    version='0.01',
    description='Bayesian Quantum Error Mitigation',
    author='Muqing Zheng',
    author_email='muz219@lehigh.edu',
    url='https://arxiv.org/abs/2010.09188',
    py_modules=['gatesimu', 'measfilter', 'QAOAfuncs', 'expfuncs'],
    data_files=[('plot', ['Plot/quantum-plots.mplstyle'])],
    )