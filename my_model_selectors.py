import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        return self

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except Exception:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)

class SelectorAIC(ModelSelector):
    """ select the model with the lowest Akaike's Information Criterion(AIC) score
    I've decided to include AIC as well since it is so similar to BIC
    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Akaike's information criteria: AIC = -2 * logL + 2p
    """

    def select(self):
        """ select the best model for self.this_word based on
        AIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score, best_model = float("inf"), None
        n_observations = self.X.shape[0]
        n_features = self.X.shape[1]
        
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            model = None
            logL = None
            try:
                model = GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=1000, random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                logL = model.score(self.X, self.lengths)
                n_parameters = n_components * (n_components - 1) + 2 * n_features * n_components
                aicValue = -2 * logL + 2 * n_parameters
                if aicValue < best_score:
                    best_score, best_model = aicValue, model
            except Exception:
                break

        if best_model is not None:
            return best_model
        else:
            return self.base_model(self.n_constant)

class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score, best_model = float("inf"), None
        n_observations = self.X.shape[0]
        n_features = self.X.shape[1]
        
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            model = None
            logL = None
            try:
                model = GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=1000, random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                logL = model.score(self.X, self.lengths)
                n_parameters = n_components * (n_components - 1) + 2 * n_features * n_components
                logN = np.log(n_observations)
                bicScore = -2 * logL + n_parameters * logN
                if bicScore < best_score:
                    best_score, best_model = bicScore, model
            except Exception:
                break

        if best_model is not None:
            return best_model
        else:
            return self.base_model(self.n_constant)

        


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score, best_model = float("-inf"), None
        
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            model = None
            logL = 0
            antiLogL = 0
            antiWordsLength = 0
            try:
                model = GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=1000, random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                logL = model.score(self.X, self.lengths)
                # Loop through ever word that's not our word to get antiLogL
                for word in self.hwords:
                    if word is not self.this_word:
                        X, lengths = self.hwords[word]
                        antiLogL += model.score(X, lengths)
                        antiWordsLength += 1
                    else:
                        continue
                # I don't subtract 1 from antiWordsLength because I have done so by NOT adding 1 when word is looped through in hwords
                dicScore = logL - 1/(antiWordsLength) * antiLogL
                
            except Exception:
                break
            if dicScore > best_score:
                best_score, best_model = dicScore, model

        if best_model is not None:
            return best_model
        else:
            return self.base_model(self.n_constant)

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''
    n_splits = 3

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        best_score, best_model = float("-inf"), None
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            scores = []
            n_splits = SelectorCV.n_splits
            model = None
            logL = None
            if(len(self.sequences) < n_splits):
                break

            split_method = KFold(random_state=self.random_state, n_splits=n_splits)
            for training_idices, testing_idices in split_method.split(self.sequences):
                x_train, lengths_train = combine_sequences(training_idices, self.sequences)
                x_test, lengths_test = combine_sequences(testing_idices, self.sequences)
                try:
                    model = GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=1000, random_state=self.random_state, verbose=False).fit(x_train, lengths_train)
                    logL = model.score(x_test, lengths_test)
                    scores.append(logL)
                except Exception:
                    break

                if len(scores) > 0:
                    avg = np.average(scores)
                if avg > best_score:
                    best_score = avg
                    best_model = model
        if best_model is not None:
            return best_model
        else:
            return self.base_model(self.n_constant)