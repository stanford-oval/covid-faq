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
import tablib
from tablib import Dataset

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

    batch = [question for question  in val_data.questions]
    topk_indices, scores, max_indices = model(batch)

    correct_classifier = 0
    correct_knn = 0
    total_in_domain = 0

    error_report = tablib.Dataset(headers=['User Query', 'Reference Question', 'kNN Precision@k', 'Classifier Topk', 'Correct'])

    for i, (question, label_strs) in enumerate(val_data):
        # shift 2 for non-ood labels
        labels = [int(n) - 2 if int(n) != -1 else -1 for n in label_strs.split(',')] 

        predict_label = topk_indices[i][max_indices[i]]
        ref_question = dataset.questions[predict_label]

        if -1 not in labels:
            retrieved = topk_indices[i].tolist()
            relevant = list(set(retrieved) & set(labels))
            precision_at_k = len(relevant) / len(retrieved)
            if relevant:
                correct_knn += 1

            if predict_label in labels:
                correct_classifier += 1
                error_report.append((question, ref_question, precision_at_k, scores[i], 1))
            else:
                error_report.append((question, ref_question, precision_at_k, scores[i], 0))

            total_in_domain += 1
        else:
            error_report.append((question, '', '', '', ''))

    print("Percentage of in domain data =", total_in_domain / len(batch))
    print("kNN Recall@K =", correct_knn / total_in_domain)
    print("Classifier accuracy =", correct_classifier / correct_knn)

    print("Accuracy (in domain) =", correct_classifier / total_in_domain)
    print("Accuracy =", correct_classifier / len(batch))

    #print(str(argv.weight_knn) + ', ' + str(argv.weight_binary_classifier) + ', ' + str(total_in_domain / len(batch)) + ', ' + str(correct_knn / total_in_domain) + ', ' + str(correct_classifier / correct_knn) + ', ' + str(correct_classifier / total_in_domain) + ', ' + str(correct_classifier / len(batch)))

    with open('data/error_report.xlsx', 'wb') as f:
        f.write(error_report.export('xlsx'))
