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

import numpy as np
import os
import io
import nsNLP

####################################################
# Functions
####################################################


# Create directory
def create_directories(output_directory, xp_name):
    """
    Create image directory
    :return:
    """
    # Directories
    image_directory = os.path.join(output_directory, xp_name, "images")
    words_directory = os.path.join(output_directory, xp_name, "words")

    # Create if does not exists
    if not os.path.exists(image_directory):
        os.mkdir(image_directory)
    # end if

    # Create if does not exists
    if not os.path.exists(words_directory):
        os.mkdir(words_directory)
    # end if

    return image_directory, words_directory
# end create_directories


####################################################
# Main function
####################################################

if __name__ == "__main__":

    # Argument parser
    args = nsNLP.tools.ArgumentBuilder(desc=u"Test a word embeddings with Echo State Network")

    # Embeddings parameters
    args.add_argument(command="--input", name="input", help="Embeddings file", type=str, required=True, extended=False)

    # Output parameters
    args.add_argument(command="--fig-size", name="fig_size", help="Figure size (pixels)", type=float, default=1024.0,
                      extended=False)
    args.add_argument(command="--min-count", name="min_count", type=int,
                      help="Minimum token count to be in the final embeddings", default=100, required=False,
                      extended=False)
    args.add_argument(command="--output", name="output", type=str, help="Experiment's output directory", required=True,
                      extended=False)

    # Parse arguments
    args.parse()

    # Questions-words benchmarks
    questions_words = nsNLP.measures.QuestionsWords()

    # Create image directory
    image_directory, words_directory = create_directories(args.output, u"")

    # Export word embeddings
    word_embeddings = nsNLP.embeddings.Embeddings()
    print(u"Word embeddings vocabulary size: {}".format(word_embeddings.voc_size))

    # Clean
    word_embeddings.clean('count', args.min_count)
    print(u"Cleaned word embeddings vocabulary size: {}".format(word_embeddings.voc_size))

    # Export image of top 100 words
    word_embeddings.wordnet('count',
                            os.path.join(image_directory, u"wordnet_TSNE.png"),
                            n_words=args.top_words,
                            fig_size=args.fig_size, reduction='TSNE')
    word_embeddings.wordnet('count',
                            os.path.join(image_directory, u"wordnet_PCA.png"),
                            n_words=args.top_words,
                            fig_size=args.fig_size, reduction='PCA')

    # Export list of words
    word_embeddings.wordlist(os.path.join(words_directory, u"wordlist.csv"))

    # Measure performance
    positioning, poss, ratio = questions_words.positioning(word_embeddings, func='inv', csv_file=os.path.join(words_directory, u"results.csv"))
    print(u"Positioning: {}".format(positioning))
    print(u"Ratio: {}".format(ratio))

    # Save positioning as a fold
    average = np.array([])
    for index, pos in enumerate(poss):
        average = np.append(average, poss)
    # end for
    print(u"Average positioning: {}".format(average))
# end if
