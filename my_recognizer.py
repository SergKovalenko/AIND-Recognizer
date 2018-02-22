import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for word_id in test_set.get_all_sequences():
        features, sequence_length = test_set.get_item_Xlengths(word_id)
        likelihoods = {}

        highest_score = float("-inf")
        best_guess = None

        for word, model in models.items():
            try:

                score = model.score(features, sequence_length)
                likelihoods[word] = score

                if score > highest_score:
                    highest_score = score
                    best_guess = word

            except:
                likelihoods[word] = float("-inf")
                continue

        guesses.append(best_guess)
        probabilities.append(likelihoods)

    return probabilities, guesses

