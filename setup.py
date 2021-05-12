#
# Copyright 2021 The Board of Trustees of the Leland Stanford Junior University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Giovanni Campagna <gcampagn@cs.stanford.edu>

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='covidfaq',
    version='0.1.0',

    packages=setuptools.find_packages(exclude=['tests']),
    entry_points= {
        'console_scripts': ['covidfaq=covidfaq.__main__:main'],
    },
    license='Apache-2.0',
    author="Stanford University Open Virtual Assistant Lab",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/stanford-oval/covid-faq",

    install_requires=[
        'torch',
        'transformers',
        'sentence-transformers',
        'tablib',
        'kfserving',
        'smart_open[s3]',
        'nltk',
        'gensim'
    ]
)
