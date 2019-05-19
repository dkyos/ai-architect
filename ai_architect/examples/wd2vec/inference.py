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
import argparse
import logging
import sys

from ai_architect.models.wd2vec import WD2vec
from ai_architect.utils.io import validate_existing_filepath, check_size

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--wd2vec_model_file',
        default='wd2vec.train.model',
        help='path to the file with the wd2vec model to load.',
        type=validate_existing_filepath)
    arg_parser.add_argument(
        '--wd',
        default='woman',
        type=str,
        action=check_size(min_size=1),
        required=True,
        help='WD to print its word vector.')

    args = arg_parser.parse_args()

    wd2vec_model = WD2vec.load(args.wd2vec_model_file)

    #print(args.wd)
    #print(wd2vec_model.wv.vocab)

    print("word vector for the WD \'" + args.wd + "\':"
        , wd2vec_model.most_similar(args.wd))
