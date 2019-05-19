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

import json
import logging
import sys

from gensim.models import FastText, Word2Vec, KeyedVectors
from gensim.models.word2vec import LineSentence
from gensim import utils
import nltk
from nltk.corpus import conll2000
from six import iteritems

logger = logging.getLogger(__name__)


# pylint: disable-msg=too-many-instance-attributes
class WD2vec:
    """
    Initialize the wd2vec model, train it, save it and load it.
    """

    # pylint: disable-msg=too-many-arguments
    # pylint: disable-msg=too-many-locals
    # pylint: disable-msg=too-many-branches
    def __init__(  # noqa: C901
            self,
            corpus,
            corpus_format='txt',
            word_embedding_type='word2vec',
            sg=0,
            size=100,
            window=10,
            alpha=0.025,
            min_alpha=0.0001,
            min_count=5,
            sample=1e-5,
            workers=20,
            hs=0,
            negative=25,
            cbow_mean=1,
            iterations=15,
            min_n=1,
            max_n=6,
            prune_non_np=True):
        """
        Initialize wd2vec model and train it.

        Args:
          corpus (str): path to the corpus.
          corpus_format (str {json,txt,conll2000}): format of the input marked corpus; txt and json
          formats are supported. For json format, the file should contain an iterable of
          sentences. Each sentence is a list of terms (unicode strings) that will be used for
          training.
          mark_char (char): special character that marks NP's suffix.
          word_embedding_type (str {word2vec,fasttext}): word embedding model type; word2vec and
          fasttext are supported.
          wd2vec_model_file (str): path to the file where the trained wd2vec model has to be
          stored.
          word_embedding_type is fasttext and word_ngrams is 1, binary should be set to True.
          sg (int {0,1}): model training hyperparameter, skip-gram. Defines the training
          algorithm. If 1, CBOW is used,otherwise, skip-gram is employed.
          size (int): model training hyperparameter, size of the feature vectors.
          window (int): model training hyperparameter, maximum distance between the current and
          predicted word within a sentence.
          alpha (float): model training hyperparameter. The initial learning rate.
          min_alpha (float): model training hyperparameter. Learning rate will linearly drop to
          `min_alpha` as training progresses.
          min_count (int): model training hyperparameter, ignore all words with total frequency
          lower than this.
          sample (float): model training hyperparameter, threshold for configuring which
          higher-frequency words are randomly downsampled, useful range is (0, 1e-5)
          workers (int): model training hyperparameter, number of worker threads.
          hs (int {0,1}): model training hyperparameter, hierarchical softmax. If set to 1,
          hierarchical softmax will be used for model training. If set to 0, and `negative` is non-
                        zero, negative sampling will be used.
          negative (int): model training hyperparameter, negative sampling. If > 0, negative
          sampling will be used, the int for negative specifies how many "noise words" should be
          drawn (usually between 5-20). If set to 0, no negative sampling is used.
          cbow_mean (int {0,1}): model training hyperparameter. If 0, use the sum of the context
          word vectors. If 1, use the mean, only applies when cbow is used.
          iterations (int): model training hyperparameter, number of iterations.
          min_n (int): fasttext training hyperparameter. Min length of char ngrams to be used
          for training word representations.
          max_n (int): fasttext training hyperparameter. Max length of char ngrams to be used for
          training word representations. Set `max_n` to be lesser than `min_n` to avoid char
          ngrams being used.
          vectors with subword (ngrams) information. If 0, this is equivalent to word2vec training.
          prune_non_np (bool): indicates whether to prune non-NP's after training process.

        """

        self.word_embedding_type = word_embedding_type
        self.sg = sg
        self.size = size
        self.window = window
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.min_count = min_count
        self.sample = sample
        self.workers = workers
        self.hs = hs
        self.negative = negative
        self.cbow_mean = cbow_mean
        self.iter = iterations
        self.min_n = min_n
        self.max_n = max_n
        self.prune_non_np = prune_non_np

        if corpus_format == 'txt':
            self._sentences = LineSentence(corpus)
            print(self._sentences)
        elif corpus_format == 'json':
            with open(corpus) as json_data:
                self._sentences = json.load(json_data)
        else:
            logger.error('invalid corpus format: %s', corpus_format)
            sys.exit(0)

        logger.info('training wd2vec model')
        self._train()

    def _train(self):
        """
        Train the wd2vec model.
        """
        if self.word_embedding_type == 'word2vec':
            self.model = Word2Vec(
                self._sentences,
                sg=self.sg,
                size=self.size,
                window=self.window,
                alpha=self.alpha,
                min_alpha=self.min_alpha,
                min_count=self.min_count,
                sample=self.sample,
                workers=self.workers,
                hs=self.hs,
                negative=self.negative,
                cbow_mean=self.cbow_mean,
                iter=self.iter)

        elif self.word_embedding_type == 'fasttext':
            self.model = FastText(
                self._sentences,
                sg=self.sg,
                size=self.size,
                window=self.window,
                alpha=self.alpha,
                min_alpha=self.min_alpha,
                min_count=self.min_count,
                sample=self.sample,
                workers=self.workers,
                hs=self.hs,
                negative=self.negative,
                cbow_mean=self.cbow_mean,
                iter=self.iter,
                min_n=self.min_n,
                max_n=self.max_n)
        else:
            logger.error('invalid word embedding type: %s', self.word_embedding_type)
            sys.exit(0)

    def save(self, wd2vec_model_file='wd2vec.model', word2vec_format=True):
        """
        Save the wd2vec model.

        Args:
            wd2vec_model_file (str): the file containing the wd2vec model to load
            word2vec_format(bool): boolean indicating whether to save the model in original
            word2vec format.
        """
        if self.word_embedding_type == 'fasttext':
            self.model.save(wd2vec_model_file)
        else:
            self.model.save(wd2vec_model_file)

    @classmethod
    def load(self, wd2vec_model_file, word2vec_format=True):
        """
        Load the wd2vec model.

        Args:
            wd2vec_model_file (str): the file containing the wd2vec model to load
            word2vec_format(bool): boolean indicating whether the model to load has been stored in
            original word2vec format.

        Returns:
            wd2vec model to load
        """
        self.model = Word2Vec.load(wd2vec_model_file)
        return self.model

