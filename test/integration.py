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

import urllib.request
import json

MODEL_URL = 'http://127.0.0.1:8080/v1/models/covid-faq:predict'

def test_basic():
    request = json.dumps({
        'instances': [
            "what are the vaccine side effects?",
            "is the vaccine free?"
        ]
    })
    with urllib.request.urlopen(MODEL_URL, data=request.encode('utf-8')) as fp:
        response = json.load(fp)

    assert isinstance(response['predictions'], list)
    assert len(response['predictions']) == 2
    assert response['predictions'][0]['answer'].startswith('Vaccine recipients commonly experience mild to moderate side effects')
    assert isinstance(response['predictions'][0]['score'], float)
    assert response['predictions'][1]['answer'].startswith('The federal government is providing the vaccine free of charge to all people')
    assert isinstance(response['predictions'][1]['score'], float)

def main():
    test_basic()

if __name__ == '__main__':
    main()