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

from typing import NamedTuple


class HParams(NamedTuple):
    nearest_neighbor_model_qq : str
    nearest_neighbor_model_qa : str
    k : int

    binary_classifier_model_qq : str
    binary_classifier_model_qa : str
    weight_tf_idf: float
    weight_knn : float
    weight_binary_classifier : float
    confidence_level : float


def parse_argv(parser):
    parser.add_argument('--knn-model', type=str, default='paraphrase-distilroberta-base-v1',
                        help="Nearest neighbor model")
    parser.add_argument('--knn-model-qa', type=str, default='msmarco-roberta-base-v3',
                        help="Nearest neighbor model (question-answer)")

    parser.add_argument('--binary-classifier-model', type=str, default='cross-encoder/stsb-roberta-large',
                        help="Cross-encoder sentence similarity model")
    parser.add_argument('--binary-classifier-model-qa', type=str, default='cross-encoder/ms-marco-MiniLM-L-12-v2',
                        help="Cross-encoder sentence similarity model (question-answer)")

    parser.add_argument('--weight-tf-idf', type=float, default=0, help="Weight of TF-IDF")
    parser.add_argument('--weight-knn', type=float, default=0.03, help="Weight of KNNs")
    parser.add_argument('--weight-binary-classifier', type=float, default=0.02, help="Weight of binary classifiers")

    parser.add_argument('--confidence', type=float, default=0.37, help="Confidence threshold")
    parser.add_argument('--top-k', type=int, default=5, help="Top-K hyperparameter for nearest neighbor")


def argv_to_hparams(argv) -> HParams:
    return HParams(nearest_neighbor_model_qq=argv.knn_model,
                   nearest_neighbor_model_qa=argv.knn_model_qa,
                   k=argv.top_k,
                   binary_classifier_model_qq=argv.binary_classifier_model,
                   binary_classifier_model_qa=argv.binary_classifier_model_qa,
                   weight_tf_idf=argv.weight_tf_idf,
                   weight_knn=argv.weight_knn,
                   weight_binary_classifier=argv.weight_binary_classifier,
                   confidence_level=argv.confidence)
