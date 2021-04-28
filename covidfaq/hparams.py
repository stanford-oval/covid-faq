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
    nearest_neighbor_model : str
    k : int

    binary_classifier_model : str
    confidence_level : float


def parse_argv(parser):
    parser.add_argument('--knn-model', type=str, default='paraphrase-distilroberta-base-v1',
                        help="Nearest neighbor model")
    parser.add_argument('--binary-classifier-model', type=str, default='cross-encoder/stsb-TinyBERT-L-4',
                        help="Cross-encoder sentence similarity model")
    parser.add_argument('--confidence', type=float, default=0.37, help="Confidence threshold")
    parser.add_argument('--top-k', type=int, default=5, help="Top-K hyperparameter for nearest neighbor")


def argv_to_hparams(argv) -> HParams:
    return HParams(nearest_neighbor_model=argv.knn_model,
                   k=argv.top_k,
                   binary_classifier_model=argv.binary_classifier_model,
                   confidence_level=argv.confidence)
