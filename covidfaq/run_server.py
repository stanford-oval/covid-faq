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

import kfserving

from .model import Model
from .data import Dataset
from . import hparams


class KFModelServer(kfserving.KFModel):
    def __init__(self, argv):
        super().__init__(argv.inference_name)
        self.argv = argv
        self.data = Dataset(self.argv.data, self.argv.faq_list)
        hp = hparams.argv_to_hparams(self.argv)
        self.model = Model(hp, self.data)
        self.ready = True

    def predict(self, request):
        topk_indices, scores, max_indices = self.model(request['instances'])
        return {
            "predictions": [
                { "answer": self.data.answers[ topk_indices[i][max_indices[i]] ], "score": float(scores[i][max_indices[i]]) }
                for i in range(len(topk_indices))
            ]
        }


def parse_argv(parser):
    parser.add_argument('-d', '--data', type=str, required=True,
                        help="Path to data directory")
    parser.add_argument('-f', '--faq-list', type=str, required=True,
                        help="File name of FAQ list")
    parser.add_argument('-n', '--inference-name', type=str, required=True,
                        help="Inference name to use for KF Serving")
    hparams.parse_argv(parser)


def main(args):
    model_server = KFModelServer(args)
    kfserving.KFServer(workers=1).start([model_server])
