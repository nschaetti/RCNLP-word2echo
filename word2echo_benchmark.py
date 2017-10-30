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

####################################################
# Main function
####################################################

if __name__ == "__main__":

    # Argument parser
    args = nsNLP.tools.ArgumentBuilder(desc=u"Word prediction with Echo State Network and one-hot vector")

    # Dataset
    args.add_argument(command="--dataset", name="dataset", type=str,
                      help="Dataset's directory", required=True, extended=False)

    # ESN arguments
    args.add_argument(command="--reservoir-size", name="reservoir_size", type=float, help="Reservoir's size",
                      required=True, extended=True)
    args.add_argument(command="--spectral-radius", name="spectral_radius", type=float, help="Spectral radius",
                      default="1.0", extended=True)
    args.add_argument(command="--leak-rate", name="leak_rate", type=str, help="Reservoir's leak rate", extended=True,
                      default="1.0")
    args.add_argument(command="--input-scaling", name="input_scaling", type=str, help="Input scaling", extended=True,
                      default="0.5")
    args.add_argument(command="--input-sparsity", name="input_sparsity", type=str, help="Input sparsity", extended=True,
                      default="0.05")
    args.add_argument(command="--w-sparsity", name="w_sparsity", type=str, help="W sparsity", extended=True,
                      default="0.05")
    args.add_argument(command="--keep-w", name="keep_w", action='store_true', help="Keep W matrix", default=False,
                      extended=False)
    args.add_argument(command="--state-gram", name="state_gram", type=str, help="State-gram value",
                      extended=True, default="1")

    # Embeddings parameters
    args.add_argument(command="--image", name="image", help="Output image", default=None, required=False,
                      extended=False)
    args.add_argument(command="--voc-size", name="voc_size", help="Vocabulary size", default=5000, required=True,
                      extended=False)
    args.add_argument(command="fig-size", name="fig_size", help="Figure size (pixels)", default=1024.0, required=False,
                      extended=False)
    args.add_argument(command="count-limit-display", name="count_limit_display",
                      help="Lower limit of word count to display a word", default=50, required=False, extended=False)

    # Experiment output parameters
    args.add_argument(command="--name", name="name", type=str, help="Experiment's name", extended=False, required=True)
    args.add_argument(command="--description", name="description", type=str, help="Experiment's description",
                      extended=False, required=True)
    args.add_argument(command="--output", name="output", type=str, help="Experiment's output directory", required=True,
                      extended=False)
    args.add_argument(command="--verbose", name="verbose", type=int, help="Verbose level", default=2, extended=False)

    # Argument
    parser.add_argument("--size", type=int, help="Max tokens to take in the dataset", default=-1)
    parser.add_argument("--loop", type=int, help="Number of loops", default=1)
    parser.add_argument("--norm", action='store_true', help="Normalize word embeddings?", default=False)
    parser.add_argument("--output", type=str, help="", default=None, required=False)
    parser.add_argument("--wordsims", type=str, help="Word similarity dataset", required=True)
    parser.add_argument("--n-similar-words", type=int, help="Number of similar words", default=20)
    args = parser.parse_args()

    # Parse arguments
    args.parse()

    # Print precision
    np.set_printoptions(precision=3)
    np.set_printoptions(threshold=np.nan)

# end if
