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
# Author: kevintangzero

import json
import tablib
from tablib import Dataset

from .data import Dataset


def parse_argv(parser):
    parser.add_argument('-d', '--data', type=str, required=True,
                        help="Path to data directory")
    parser.add_argument('-f', '--format', type=str, required=True, choices=['json', 'xls'])
    parser.add_argument('-o', '--output', type=str, required=True)


def main(argv):
    dataset = Dataset(argv.data)

    data = tablib.Dataset(headers=['Question', 'Answer'])
    for q, a in dataset:
        data.append((q, a))

    if argv.format == 'json':
        with open(argv.output, 'w') as f:
            json.dump(data.export('json'), f, indent=4)
    else:
        with open(argv.output, 'wb') as f:
            f.write(data.export('xls'))

