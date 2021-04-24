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

import os
from typing import List
from smart_open import open

class Dataset:
    datadir : str
    questions : List[str]
    answers : List[str]

    def __init__(self, datadir):
        self.datadir = datadir
        self.questions = self._read_questions()
        self.answers = self._read_answers()
        assert len(self.questions) == len(self.answers)

    def __iter__(self):
        return zip(self.questions, self.answers)

    def _read_questions(self):
        questions = []
        with open(self.datadir + '/questions', 'r') as fp:
            for line in fp:
                questions.append(line.strip())
        return questions

    def _read_answers(self):
        answers = []
        answer = ''
        with open(self.datadir + '/answers', 'r') as fp:
            for line in fp:
                if line == '---\n':
                    answers.append(answer)
                    answer = ''
                    continue
                answer += line
            if answer:
                answers.append(answer)
        return answers
