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
import csv

class Dataset:
    datadir : str
    questions : List[str]
    answers : List[str]

    def __init__(self, datadir, datafile):
        self.datadir = datadir
        self.datafile = datafile
        self.questions, self.answers = self._read_faq_list()

    def __iter__(self):
        return zip(self.questions, self.answers)

    def __getitem__(self, i):
        return self.questions[i], self.answers[i]

    def _read_faq_list(self):
        questions = []
        answers = []
        with open(self.datadir + '/' + self.datafile, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                questions.append(row['Question'])
                answers.append(row['Answer'])
        assert(len(questions) == len(answers))
        return questions, answers
