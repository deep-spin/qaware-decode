# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

setup(
    name="qaware-decode",
    version="0.0.1",
    author="Patrick Fernandes & Antonio Farinhas",
    author_email="pfernand@cs.cmu.edu",
    url="https://github.com/CoderPat/qaware-decode/",
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.7",
    setup_requires=[],
    install_requires=[
        'unbabel-comet @ git+https://github.com/CoderPat/COMET.git@master',
        'sacrebleu @ git+https://github.com/mjpost/sacrebleu@master',
        'sacremoses'
    ],
    entry_points={
        'console_scripts': [
            'qaware-mbr=qaware_decode.mbr:main',
            'qaware-rerank=qaware_decode.rerank:main',
        ]
    },
    extras_require = {
        'mbart-qe': ['mbart-qe @ git+https://github.com/Unbabel/wmt21-qe-task.git@MBART-without-uncertainty'],
        'transquest': ['transquest']
    }
)