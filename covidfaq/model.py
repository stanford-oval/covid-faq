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
from typing import NamedTuple, Union, List

from .data import Dataset
from .hparams import HParams


class Answer(NamedTuple):
    answer: Union[str, None]
    score: float


class Model(torch.nn.Module):
    def __init__(self, hparams: HParams, dataset: Dataset):
        super().__init__()
        self.hparams = hparams
        self.dataset = dataset

        # load model
        self.model = SentenceTransformer(hparams.nearest_neighbor_model)
        self.cross_encoder = CrossEncoder(hparams.binary_classifier_model)

        # generate embeddings for questions
        self.embeddings_q = self.model.encode(self.dataset.questions)

    def forward(self, query_batch: List[str]) -> List[Answer]:
        embedding = self.model.encode(query_batch)

        cosine_scores = util.pytorch_cos_sim(embedding, self.embeddings_q)
        topk_scores, topk_indices = torch.topk(cosine_scores, self.hparams.k)
        # print(answers[topk_indices[0][0]])
        # print(topk_scores[0][0])

        batch = []
        for i, query in enumerate(query_batch):
            for j in range(self.hparams.k):
                batch.append((query, self.dataset.questions[topk_indices[i][j]]))
        scores = self.cross_encoder.predict(batch)
        scores = np.reshape(scores, (len(query_batch), self.hparams.k))
        # print(scores)

        max_scores = np.max(scores, axis=1)
        max_indices = np.argmax(scores, axis=1)
        answers = []
        for i in range(len(query_batch)):
            max_score = max_scores[i]
            max_index = max_indices[i]
            if max_score > self.hparams.confidence_level:
                ans = self.dataset.answers[topk_indices[i][max_index]]
            else:
                ans = None
            answers.append(Answer(answer=ans, score=float(max_score)))

        return answers, topk_indices, scores, max_indices
