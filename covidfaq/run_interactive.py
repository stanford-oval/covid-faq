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

import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder
import configparser

from . import hparams
from .data import Dataset
from .model import Model

def parse_argv(parser):
    parser.add_argument('-d', '--data', type=str, required=True,
                        help="Path to data directory")
    parser.add_argument('-f', '--faq-list', type=str, required=True,
                        help="File name of FAQ list")
    hparams.parse_argv(parser)


def main(argv):
    dataset = Dataset(argv.data, argv.faq_list)
    hp = hparams.argv_to_hparams(argv)
    model = Model(hp, dataset)

    # interactive query
    while True:
        try:
            query = input('> ')
            answers, topk_indices, scores = model([query])
            ans, score = answers[0]
            if ans is not None:
                print(ans)
                print('Score =', score)
            else:
                print("Sorry! We don't know the answer now. Please come back later.")
                print('Score =', score)
        except (EOFError, KeyboardInterrupt):
            break
