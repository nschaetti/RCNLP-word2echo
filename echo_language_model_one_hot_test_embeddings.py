#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File : core.classifiers.RCNLPTextClassifier.py
# Description : Echo State Network for text classification.
# Auteur : Nils Schaetti <nils.schaetti@unine.ch>
# Date : 01.02.2017 17:59:05
# Lieu : Nyon, Suisse
#
# This file is part of the Reservoir Computing NLP Project.
# The Reservoir Computing Memory Project is a set of free software:
# you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Foobar is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
#

import argparse
import numpy as np
import logging
import pickle
from embeddings.Wordsim353 import Wordsim353
from evaluation.Metrics import Metrics
from tools.Visualization import Visualization

#########################################################################
# Experience settings
#########################################################################

# Exp. info
ex_name = "Echo Word Prediction Experience"
ex_instance = "Echo Language Model One Hot Test"

####################################################
# Main function
####################################################

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="RCNLP - Word prediction with Echo State Network and one-hot vector on Wikipedia")

    # Argument
    parser.add_argument("--file", type=str, help="Word embedding file", required=True)
    parser.add_argument("--image", type=str, help="Output image", default=None, required=False)
    parser.add_argument("--log-level", type=int, help="Log level", default=20)
    parser.add_argument("--fig-size", type=float, help="Figure size (pixels)", default=1024.0)
    parser.add_argument("--count-limit", type=int, help="Lower limit of word count to display a word", default=50)
    parser.add_argument("--wordsims", type=str, help="Word similarity dataset", required=True)
    parser.add_argument("--n-similar-words", type=int, help="Number of similar words", default=20)
    args = parser.parse_args()

    # Init logging
    logging.basicConfig(level=args.log_level, format='%(asctime)s :: %(levelname)s :: %(message)s')

    # Print precision
    np.set_printoptions(precision=3)
    np.set_printoptions(threshold=np.nan)

    # Load Wordsim353
    wordsim353 = Wordsim353.load(args.wordsims)

    # Current word embeddings
    word2vec = pickle.load(open(args.file, 'r'))
    word_embeddings = word2vec.get_word_embeddings()

    # Word embedding matrix's size
    logging.info(u"Word embedding matrix's size : {}".format(word_embeddings.shape))
    logging.info(u"Word embedding vectors average : {}".format(np.average(word_embeddings)))
    logging.info(u"Word embedding vectors sddev : {}".format(np.std(word_embeddings)))

    # For each distance measure
    for distance_measure in ['euclidian', 'cosine', 'cosine_abs']:
        print(u"#" * 100)
        print(u"# " + distance_measure)
        print(u"#" * 100)

        # Similarities
        Visualization.similar_words(
            [u"he", u"computer", u"million", u"Toronto", u"France", u"phone", u"ask", u"september", u"blue", u"king",
             u"man", u"woman"],
            word2vec, distance_measure=distance_measure, limit=args.n_similar_words)

        # Word computing
        Visualization.king_man_woman(word2vec, u"king", u"man", u"woman", distance_measure=distance_measure)

        # Test relatedness
        relatedness, relatedness_words = Metrics.relatedness(wordsim353, word2vec, distance_measure=distance_measure)
        print(u"Relatedness : {}, on {} words".format(relatedness, relatedness_words))
    # end for

    # If we want a figure
    if args.image is not None:
        selected_words = [u"switzerland", u"france", u"italy", u"spain", u"germany", u"canada", u"belgium", u"bern",
                          u"paris", u"rome", u"madrid", u"berlin", u"ottawa", u"brussels"]
        Visualization.top_words_figure(word2vec, word_embeddings, args.image, args.fig_size, args.count_limit)
        Visualization.words_figure(selected_words, word2vec, word_embeddings, args.image + u"_words", args.fig_size,
                                   reduction='PCA')
    # end if

# end if
