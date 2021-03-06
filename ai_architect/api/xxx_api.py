# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

from ai_architect.api.abstract_api import AbstractApi


class XxxApi(AbstractApi):
    """
    XXX API
    """
    def __init__(self):
        self.model = None

    def load_model(self):
        """
        Load XXX model
        """
        print("Load XXX model")
        #######

    def inference(self, doc):
        """
        XXXX model

        Args:
            doc (str): the doc str

        Returns:
            XXXX
        """

        print("Inference XXX model")
        print("Doc: " + doc)
        ret = {'doc_text': doc, 'annotation_set': []}
        spans = []
        available_tags = set()
        ret['annotation_set'] = list(available_tags)
        ret['spans'] = spans
        ret['title'] = 'None'

        print({"doc": ret, 'type': 'high_level'})

        return {'doc': ret, 'type': 'high_level'}
        #return self.model.parse(doc)
