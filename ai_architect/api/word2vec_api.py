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
import pickle
from os import makedirs, path, sys

import numpy as np

from ai_architect.api.abstract_api import AbstractApi
from ai_architect.models.wd2vec import WD2vec
from ai_architect import LIBRARY_OUT
from ai_architect.utils.generic import pad_sentences
from ai_architect.utils.io import download_unlicensed_file



class Word2VecApi(AbstractApi):
    """
    Word2Vec API
    """
    model_dir = str(LIBRARY_OUT / 'wd2vec-pretrained')
    pretrained_model = path.join(model_dir, 'wd2vec_v0.h5')

    def __init__(self):
        print("Word2VecApi: init")
        self.model = None

    def load_model(self):
        """
        Load Word2Vec model
        """
        print("Word2VecApi: load_model")
        self.model = WD2vec.load(self.pretrained_model)

    def inference(self, doc):
        """
        XXXX model

        Args:
            doc (str): the doc str

        Returns:
            XXXX

        """
        print("Word2VecApi: inference")
        print("Doc: " + doc)
        ret_str = str(self.model.most_similar(doc))
        ret = {'doc_text': ret_str , 'annotation_set': []}
        spans = []
        available_tags = set()
        ret['annotation_set'] = list(available_tags)
        ret['spans'] = spans
        ret['title'] = 'None'

        print({"doc": ret, 'type': 'high_level'})

        return {'doc': ret, 'type': 'high_level'}
