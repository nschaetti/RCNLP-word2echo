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
import os
import io
from embeddings.Word2Vec import Word2Vec, OneHotVectorFullException
from embeddings.EchoWordPrediction import EchoWordPrediction
from scipy.spatial.distance import euclidean
from sklearn.manifold import TSNE
import pylab as plt
import logging
import pickle
from embeddings.Wordsim353 import Wordsim353
from evaluation.Metrics import Metrics

#########################################################################
# Experience settings
#########################################################################

# Exp. info
ex_name = "Echo Word Prediction Experience"
ex_instance = "Echo Language Model One Hot"

# Reservoir Properties
rc_leak_rate = 0.5  # Leak rate
rc_input_scaling = 1.0  # Input scaling
rc_size = 500  # Reservoir size
rc_spectral_radius = 0.9  # Spectral radius
rc_w_sparsity = 0.1
rc_input_sparsity = 0.01

####################################################
# Functions
####################################################

####################################################
# Main function
####################################################

if __name__ == "__main__":

    # Argument parser
    parser = argparse.ArgumentParser(description="RCNLP - Word prediction with Echo State Network and one-hot vector")

    # Argument
    parser.add_argument("--dataset", type=str, help="Dataset's directory", required=True)
    parser.add_argument("--image", type=str, help="Output image", default=None, required=False)
    parser.add_argument("--size", type=int, help="How many file to take in the dataset", default=-1)
    parser.add_argument("--sparse", action='store_true', help="Sparse matrix?", default=False)
    parser.add_argument("--log-level", type=int, help="Log level", default=20)
    parser.add_argument("--voc-size", type=int, help="Vocabulary size", default=5000, required=True)
    parser.add_argument("--loop", type=int, help="Number of loops", default=1)
    parser.add_argument("--fig-size", type=float, help="Figure size (pixels)", default=1024.0)
    parser.add_argument("--count-limit", type=int, help="Lower limit of word count to display a word", default=50)
    parser.add_argument("--norm", action='store_true', help="Normalize word embeddings?", default=False)
    parser.add_argument("--output", type=str, help="", default=None, required=False)
    parser.add_argument("--wordsims", type=str, help="Word similarity dataset", required=True)
    args = parser.parse_args()

    # Init logging
    logging.basicConfig(level=args.log_level)
    logger = logging.getLogger(name="RCNLP")

    # Print precision
    np.set_printoptions(precision=3)
    np.set_printoptions(threshold=np.nan)

    # Load Wordsim353
    wordsim353 = Wordsim353.load(args.wordsims)

    # Word2Vec
    word2vec = Word2Vec(dim=args.voc_size, mapper='one-hot')

    # ESN for word prediction
    esn_word_prediction = EchoWordPrediction(word2vec=word2vec, size=rc_size, leaky_rate=rc_leak_rate,
                                             spectral_radius=rc_spectral_radius, input_scaling=rc_input_scaling,
                                             input_sparsity=rc_input_sparsity, w_sparsity=rc_w_sparsity,
                                             use_sparse_matrix=args.sparse)

    # Current word embeddings
    word_embeddings = None
    last_word_embeddings = None

    # For each loop
    for loop in range(args.loop):
        # Change W_in
        if word_embeddings is not None:
            last_word_embeddings = word_embeddings
            esn_word_prediction.set_w_in(word_embeddings[:-1, :])
        # end if

        # Add text examples
        for index, file in enumerate(os.listdir(args.dataset)):
            if args.size != -1 and index >= args.size:
                break
            # end if
            file_path = os.path.join(args.dataset, file)
            logger.info(u"Adding text file {}/{} : {}".format(index+1, args.size, file_path))
            try:
                esn_word_prediction.add(io.open(file_path, 'r').read())
            except OneHotVectorFullException:
                logger.warning(u"One-hot vector representation is full!")
                break
                pass
            # end try
        # end for

        # Word counts and voc size
        logger.info(u"Vocabulary size : {}".format(word2vec.get_n_words()))
        logger.info(u"Number of tokens : {}".format(word2vec.get_total_count()))

        # Train
        logger.info(u"Training...")
        esn_word_prediction.train()

        # Get word embeddings
        word_embeddings = esn_word_prediction.get_word_embeddings()

        # Word embedding matrix's size
        logger.info(u"Word embedding matrix's size : {}".format(word_embeddings.shape))
        logger.info(u"Word embedding vectors average : {}".format(np.average(word_embeddings)))
        logger.info(u"Word embedding vectors sddev : {}".format(np.std(word_embeddings)))

        # Normalize word embeddings
        if args.norm:
            word_embeddings -= np.average(word_embeddings)
            word_embeddings /= np.std(word_embeddings)
            logger.info(u"Normalized word embedding vectors average : {}".format(np.average(word_embeddings)))
            logger.info(u"Normalized word embedding vectors sddev : {}".format(np.std(word_embeddings)))
        # end if

        # Set word embeddings
        word2vec.set_word_embeddings(word_embeddings=word_embeddings)

        # Distance with preceding word embeddings
        if last_word_embeddings is not None:
            average_distance = 0.0
            for i in range(args.voc_size):
                average_distance += euclidean(word_embeddings[:, i], last_word_embeddings[:, i])
            # end for
            logger.info(u"Distance with preceding word embeddings : {}".format(average_distance / float(args.voc_size)))
        # end if

        # Similarities
        logger.info(u"Words similar to he ({}) : {}".format(word2vec.get_word_count(u"he"), word2vec.get_similar_words(u"he")))
        logger.info(u"Words similar to computer ({}) : {}".format(word2vec.get_word_count(u"computer"), word2vec.get_similar_words(u"computer")))
        logger.info(u"Words similar to million ({}) : {}".format(word2vec.get_word_count(u"million"), word2vec.get_similar_words(u"million")))
        logger.info(u"Words similar to Toronto ({}) : {}".format(word2vec.get_word_count(u"Toronto"), word2vec.get_similar_words(u"Toronto")))
        logger.info(u"Words similar to France ({}) : {}".format(word2vec.get_word_count(u"France"), word2vec.get_similar_words(u"France")))
        logger.info(u"Words similar to phone ({}) : {}".format(word2vec.get_word_count(u"phone"), word2vec.get_similar_words(u"phone")))
        logger.info(u"Words similar to ask ({}) : {}".format(word2vec.get_word_count(u"ask"), word2vec.get_similar_words(u"ask")))
        logger.info(u"Words similar to september ({}) : {}".format(word2vec.get_word_count(u"september"), word2vec.get_similar_words(u"september")))
        logger.info(u"Words similar to blue ({}) : {}".format(word2vec.get_word_count(u"blue"), word2vec.get_similar_words(u"blue")))

        # Test relatedness
        relatedness, relatedness_words = Metrics.relatedness(wordsim353, word2vec)
        print(u"Relatedness : {}, on {} words".format(relatedness, relatedness_words))

        # If we want a figure
        if args.image is not None:
            # Order by word count
            word_counters = list()
            word_counts = word2vec.get_word_counts()
            for word_text in word_counts.keys():
                word_counters.append((word_text, word_counts[word_text]))
            # end for
            word_counters = sorted(word_counters, key=lambda tup: tup[1], reverse=True)

            # Select top-words
            selected_word_embeddings = np.zeros((501, args.count_limit))
            selected_word_indexes = dict()
            word_pos = 0
            for (word_text, word_count) in word_counters[: args.count_limit]:
                word_index = word2vec.get_word_index(word_text)
                selected_word_embeddings[:, word_pos] = word_embeddings[:, word_index]
                selected_word_indexes[word_text] = word_pos
                word_pos += 1
            # end for

            # Word embedding matrix's size
            logger.info(u"Selected word embeddings matrix's size : {}".format(selected_word_embeddings.shape))

            # Reduce with t-SNE
            logger.info(u"Reducing word embedding with TSNE")
            model = TSNE(n_components=2, random_state=0)
            reduced_matrix = model.fit_transform(selected_word_embeddings.T)

            # Word embedding matrix's size
            logger.info(u"Reduced matrix's size : {}".format(reduced_matrix.shape))

            # Show t-SNE
            plt.figure(figsize=(args.fig_size*0.003, args.fig_size*0.003), dpi=300)
            max_x = np.amax(reduced_matrix, axis=0)[0]
            max_y = np.amax(reduced_matrix, axis=0)[1]
            min_x = np.amin(reduced_matrix, axis=0)[0]
            min_y = np.amin(reduced_matrix, axis=0)[1]
            plt.xlim((min_x * 1.2, max_x * 1.2))
            plt.ylim((min_y * 1.2, max_y * 1.2))
            for word_text in selected_word_indexes.keys():
                word_count = word2vec.get_word_count(word_text)
                word_index = selected_word_indexes[word_text]
                plt.scatter(reduced_matrix[word_index, 0], reduced_matrix[word_index, 1], 0.5)
                plt.text(reduced_matrix[word_index, 0], reduced_matrix[word_index, 1], word_text + u" (" + str(word_count) + u")", fontsize=2.5)
            # end for

            # Save image
            logger.info(u"Saving figure to {}".format(args.image + str(loop) + ".png"))
            plt.savefig(args.image + str(loop) + ".png")
        # end if

        # Save word embeddings
        if args.output is not None:
            logger.info(u"Saving word embeddings to {}".format(args.output))
            pickle.dump((word2vec.get_word_indexes(), word_embeddings), open(args.output, 'wb'))
        # end if

        # Reset word prediction
        word2vec.reset_word_count()
        esn_word_prediction.reset()
    # end if

# end if
