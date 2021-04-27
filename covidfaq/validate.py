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
    parser.add_argument('-v', '--validation', type=str, required=True,
                        help="File name of validation data")
    hparams.parse_argv(parser)


def main(argv):
    dataset = Dataset(argv.data, argv.faq_list)
    val_data = Dataset(argv.data, argv.validation)
    hp = hparams.argv_to_hparams(argv)
    model = Model(hp, dataset)

    batch = [q for q, labels in val_data]
    answers = model(batch)

    correct = 0
    for (ans, score), (q, labels) in zip(answers, val_data):
        label_nums = [int(n) for n in labels.split(',')]
        if ans is not None:
            valid_answers = [ dataset[n - 2][1] for n in label_nums if n != -1 ]
            if ans in valid_answers:
                correct += 1
        else:
            if -1 in label_nums:
                correct += 1

    print("Accuracy =", correct / len(batch))
