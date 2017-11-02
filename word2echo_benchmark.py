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


# Do we have to change W for this property
def change_w(params):
    """
    Do we have to change W for this parameter
    :param params:
    :return:
    """
    for param in params:
        if param in [u"reservoir_size", u"w_sparsity"]:
            return True
        # end if
    # end for
    return False
# end keep_w


# Get changed params
def get_changed_params(new_space, last_space):
    """
    Get changed param
    :param new_space:
    :param last_space:
    :return:
    """
    # Empty last space
    if len(last_space.keys()) == 0:
        return new_space.keys()
    # end if

    # Changed params
    changed_params = list()

    # For each param in new space
    for new_param in new_space.keys():
        if new_param not in last_space.keys():
            changed_params.append(new_param)
        else:
            if new_space[new_param] != last_space[new_param]:
                changed_params.append(new_param)
            # end if
        # end if
    # end for

    return changed_params
# end get_changed_params


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
    args = nsNLP.tools.ArgumentBuilder(desc=u"Word prediction with Echo State Network and one-hot vector")

    # Dataset
    args.add_argument(command="--dataset", name="dataset", type=str,
                      help="Dataset's directory", required=True, extended=False)
    args.add_argument(command="--dataset-size", name="dataset_size", help="Dataset size in token count", required=False,
                      extended=False, default=-1)

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
    args.add_argument(command="--w2e-model", name="w2e_model", type=str, help="Word2Echo model (word2echo, echo2word)",
                      extended=True, default="word2echo")
    args.add_argument(command="--model-direction", name="model_direction", type=str,
                      help="Model direction (lr, rl, both)",
                      extended=True, default="both")

    # Embeddings parameters
    args.add_argument(command="--voc-size", name="voc_size", help="Vocabulary size", type=int, default=5000,
                      required=True,
                      extended=False)
    args.add_argument(command="--fig-size", name="fig_size", help="Figure size (pixels)", type=float, default=1024.0,
                      extended=False)
    args.add_argument(command="--top-words", name="top_words", type=int,
                      help="Number of top words (by count) to display in the wordnet picture", default=100, required=False, extended=False)
    args.add_argument(command="--min-count", name="min_count", type=int,
                      help="Minimum token count to be in the final embeddings", default=100, required=False,
                      extended=False)

    # Experiment output parameters
    args.add_argument(command="--name", name="name", type=str, help="Experiment's name", extended=False, required=True)
    args.add_argument(command="--description", name="description", type=str, help="Experiment's description",
                      extended=False, required=True)
    args.add_argument(command="--output", name="output", type=str, help="Experiment's output directory", required=True,
                      extended=False)
    args.add_argument(command="--n-samples", name="n_samples", type=int, help="Number of different reservoir to test",
                      default=1, extended=False)
    args.add_argument(command="--verbose", name="verbose", type=int, help="Verbose level", default=2, extended=False)

    # Parse arguments
    args.parse()

    # Print precision
    np.set_printoptions(precision=3)
    np.set_printoptions(threshold=np.nan)

    # Questions-words benchmarks
    questions_words = nsNLP.measures.QuestionsWords()

    # Parameter space
    param_space = nsNLP.tools.ParameterSpace(args.get_space())

    # Experiment
    xp = nsNLP.tools.ResultManager\
    (
        args.output,
        args.name,
        args.description,
        args.get_space(),
        args.n_samples,
        questions_words.size,
        verbose=args.verbose,
        nan=True
    )

    # Create image directory
    image_directory, words_directory = create_directories(args.output, args.name)

    # W index
    w_index = 0

    # Last space
    last_space = dict()

    # Iterate
    for space in param_space:
        # Params
        reservoir_size = int(space['reservoir_size'])
        w_sparsity = space['w_sparsity']
        leak_rate = space['leak_rate']
        input_scaling = space['input_scaling']
        input_sparsity = space['input_sparsity']
        spectral_radius = space['spectral_radius']
        state_gram = space['state_gram']
        model_type = space['w2e_model'][0][0]
        model_direction = space['model_direction'][0][0]

        # Choose the right tokenizer
        tokenizer = nsNLP.tokenization.NLTKTokenizer()

        # Set experience state
        xp.set_state(space)

        # Average sample
        average_sample = np.array([])

        # For each sample
        for n in range(args.n_samples):
            # Changed parameter
            changed_params = get_changed_params(space, last_space)

            # Description
            desc_info = u"{}-{}".format(space, n)

            # Generate a new W if necessary
            if change_w(changed_params) or not args.keep_w:
                xp.write(u"\t\tGenerating new W matrix", log_level=2)
                w = nsNLP.esn_models.ESNTextClassifier.w(rc_size=reservoir_size, rc_w_sparsity=w_sparsity)
                xp.save_object(u"w_{}".format(w_index), w, info=desc_info)
            # end if

            # Set sample
            xp.set_sample_state(n)

            # Create ESN text classifier
            word2echo_model = nsNLP.esn_models.Word2Echo.create\
            (
                rc_size=reservoir_size,
                rc_spectral_radius=spectral_radius,
                rc_leak_rate=leak_rate,
                rc_input_scaling=input_scaling,
                rc_input_sparsity=input_sparsity,
                rc_w_sparsity=w_sparsity,
                w=w,
                voc_size=args.voc_size,
                state_gram=state_gram,
                model_type=model_type,
                direction=model_direction
            )

            # For each directory
            cont_add = True
            token_count = 0
            for filename in os.listdir(args.dataset):
                # File path
                file_path = os.path.join(args.dataset, filename)

                # Directory path
                xp.write(u"\t\t\tAdding file {}".format(file_path), log_level=3)

                # Open file
                text_content = io.open(file_path, 'r', encoding='utf-8').read()

                # Try to add
                try:
                    word2echo_model.add(tokenizer(text_content))
                except nsNLP.esn_models.converters.OneHotVectorFullException:
                    xp.write(u"\t\t\tOne-hot vector representation is full!", log_level=3)
                    cont_add = False
                    break
                    pass
                # end try

                # Display
                xp.write(u"\t\t\t\tVocabulary size : {}".format(word2echo_model.voc_size), log_level=4)
                xp.write(u"\t\t\t\tNumber of tokens : {}".format(word2echo_model.token_count), log_level=4)

                # Count tokens
                if args.dataset_size != -1 and word2echo_model.token_count > args.dataset_size:
                    cont_add = False
                # end if

                # Continue
                if not cont_add:
                    break
                # end if
            # end for

            # Word counts and voc size
            xp.write(u"\t\t\t\tFinal vocabulary size : {}".format(word2echo_model.voc_size), log_level=4)
            xp.write(u"\t\t\t\tFinal number of tokens : {}".format(word2echo_model.token_count), log_level=4)

            # Extract word embeddings
            xp.write(u"\t\t\tExtracting word embeddings", log_level=3)
            word2echo_model.extract()

            # Export word embeddings
            word_embeddings = word2echo_model.export_embeddings()
            xp.write(u"\t\t\tWord embeddings vocabulary size: {}".format(word_embeddings.voc_size), log_level=3)

            # Save word embeddings object
            xp.save_object(u"word_embeddings" + unicode(w_index), word_embeddings, desc_info)

            # Clean
            word_embeddings.clean('count', args.min_count)

            # Export image of top 100 words
            word_embeddings.wordnet('count',
                                    os.path.join(image_directory, u"wordnet_TSNE_" + unicode(w_index) + u".png"),
                                    n_words=args.top_words,
                                    fig_size=args.fig_size, reduction='TSNE', info=desc_info)
            word_embeddings.wordnet('count',
                                    os.path.join(image_directory, u"wordnet_PCA_" + unicode(w_index) + u".png"),
                                    n_words=args.top_words,
                                    fig_size=args.fig_size, reduction='PCA', info=desc_info)

            # Export list of words
            word_embeddings.wordlist(os.path.join(words_directory, u"wordlist" + unicode(w_index) + u".csv"))

            # Measure performance
            positioning, poss, ratio = questions_words.positioning(word_embeddings, func='inv')
            xp.write(u"\t\t\tPositioning: {}".format(positioning), log_level=3)
            #xp.write(u"\t\t\tPositionings: {}".format(poss), log_level=3)
            xp.write(u"\t\t\tRatio: {}".format(ratio), log_level=3)

            # Save positioning as a fold
            for index, pos in enumerate(poss):
                xp.set_fold_state(index)
                if index == len(poss)-1:
                    xp.add_result(pos, last_fold=True)
                else:
                    xp.add_result(pos)
                # end if
            # end for

            # Last space
            last_space = space

            # Delete classifier
            del word2echo_model

            # W index
            w_index += 1
        # end for
    # end for

    # Save experiment results
    xp.save()
# end if
