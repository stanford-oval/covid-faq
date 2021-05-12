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
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import gensim
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

        # pre-process data for tf-idf
        questions = [[w.lower() for w in word_tokenize(question)]
                                    for question in self.dataset.questions]
        self.dictionary = gensim.corpora.Dictionary(questions)
        corpus = [self.dictionary.doc2bow(question) for question in questions]

        # tf-idf
        self.tf_idf = gensim.models.TfidfModel(corpus)
        self.sims = gensim.similarities.MatrixSimilarity(self.tf_idf[corpus], num_features=len(self.dictionary))

        # load model
        self.model_qq = SentenceTransformer(hparams.nearest_neighbor_model_qq)
        self.model_qa = SentenceTransformer(hparams.nearest_neighbor_model_qa)
        self.cross_encoder_qq = CrossEncoder(hparams.binary_classifier_model_qq)
        self.cross_encoder_qa = CrossEncoder(hparams.binary_classifier_model_qa)

        # generate embeddings for questions/answers
        self.embeddings_q = self.model_qq.encode(self.dataset.questions)
        self.embeddings_a = self.model_qa.encode(self.dataset.answers)

    def forward(self, query_batch: List[str]):# -> List[Answer]:
        # tf-idf
        query_docs = [[w.lower() for w in word_tokenize(query)] for query in query_batch]
        query_bows = [self.dictionary.doc2bow(query_doc) for query_doc in query_docs]
        scores_tf_idf = self.sims[self.tf_idf[query_bows]]

        # q-q, q-a models
        embedding_qq = self.model_qq.encode(query_batch)
        embedding_qa = self.model_qa.encode(query_batch)

        cosine_scores_qq = util.pytorch_cos_sim(embedding_qq, self.embeddings_q)
        cosine_scores_qa = util.pytorch_cos_sim(embedding_qa, self.embeddings_a)
        scores_qq = (cosine_scores_qq + 1) / 2
        scores_qa = (cosine_scores_qa + 1) / 2
        #cosine_scores = (1-self.hparams.weight_knn) * cosine_scores_qq + self.hparams.weight_knn * cosine_scores_qa
        scores = (1-self.hparams.weight_knn) * scores_qq + self.hparams.weight_knn * scores_qa + self.hparams.weight_tf_idf * scores_tf_idf

        #topk_scores, topk_indices = torch.topk(cosine_scores, self.hparams.k)
        topk_scores, topk_indices = torch.topk(scores, self.hparams.k)

        batch_qq = []
        batch_qa = []
        for i, query in enumerate(query_batch):
            for j in range(self.hparams.k):
                batch_qq.append((query, self.dataset.questions[topk_indices[i][j]]))
                batch_qa.append((query, self.dataset.answers[topk_indices[i][j]]))
        scores_qq = self.cross_encoder_qq.predict(batch_qq)
        scores_qa = self.cross_encoder_qa.predict(batch_qa)
        scores = (1-self.hparams.weight_binary_classifier) * scores_qq + self.hparams.weight_binary_classifier * scores_qa
        scores = np.reshape(scores, (len(query_batch), self.hparams.k))

        #max_scores = np.max(scores, axis=1)
        max_indices = np.argmax(scores, axis=1)
        #answers = []
        #for i in range(len(query_batch)):
        #    max_score = max_scores[i]
        #    max_index = max_indices[i]
        #    if max_score > self.hparams.confidence_level:
        #        ans = self.dataset.answers[topk_indices[i][max_index]]
        #    else:
        #        ans = None
        #    answers.append(Answer(answer=ans, score=float(max_score)))

        return topk_indices, scores, max_indices

        #return answers, topk_indices, scores, max_indices
