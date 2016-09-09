
import os
import random
import time
import operator
from copy import deepcopy
from subprocess import check_output
from shlex import split as parse

import numpy as np
from scipy.stats import norm

CONTINUOUS = 2
CATEGORICAL = 3


def save_data(filename, data):
    with open(filename, 'w') as new_file:
        new_file.write(data)


def read_data(filename):
    with open(filename, 'r') as data:
        return data.read()


def cond_line(line):
    return 1 if line.startswith('conds') else 0


def remove_first_line(line):
    return line.split('\n', 1)[1]


def pairwise(iterable):
    iterator = iter(iterable)
    paired = zip(iterator, iterator)
    return paired


def arguments(p):
    arguments = [c.split('=', 1)[1] for c in parse(p)]
    return arguments


def mean(numbers):
    mean = float(sum(numbers)) / max(len(numbers), 1)
    return mean


def variance_with_mean(mean):

    def variance(numbers):
        variance = sum(map(lambda n: (n - mean)**2, numbers))
        return variance

    return variance


def parse_float_array(arraystring):
    array = [float(n) for n in arraystring.split(',')]
    return array


class Cubist:
    """
    This class wraps the cubist command line tools in a scikit-learn interface.
    The learning phase relies on the cubist command line tools, whereas the
    predictions themselves are executed directly in python.
    """

    def __init__(self, name='temp', print_output=False, unbiased=True,
                 max_rules=None, committee_members=20, max_categories=5000,
                 feature_type=None):
        """ Instantiate the cubist class with a number of invocation parameters

        Parameters
        ----------
        name: String
            The prefix of the output files (extra data is appended),
            these files should be removed automatically during training
            and after testing.
        print_output: boolean
            If true, print cubist's stdout direclty to the python console.
        unbiased: boolean
            If true, ask cubist to generate unbiased models
        max_rules: int | None
            If none, cubist can generate as many rules as it sees fit,
            otherwise; an integer limit can be added with this parameter.
        committee_members: int
            The number of cubist models to generate. Committee models can
            greatly reduce the result variance, so this should be used
            whenever possible.
        max_categories: int
            The maximum number of categories cubist will search for in the
            data when creating a categorical variable.
        feature_type:  numpy array
            An array of length equal to the number of features, 0 if
            that feature is continuous and 1 if it is categorical.
        """

        # Setting up the user details
        self._trained = False
        self._models = []
        self._filename = name + str(time.time()) + str(random.random())

        # Setting the user options
        self.print_output = print_output
        self.committee_members = committee_members
        self.unbiased = unbiased
        self.max_rules = max_rules
        self.feature_type = feature_type
        self.max_categories = max_categories

    def fit(self, x, y):
        """ Train the Cubist model
        Given a matrix of values (X) and an output vector of values (y), this
        method will train the cubist model and then read the training files
        directly as parameters of this class.

        Parameters
        ----------
        x: numpy.array
            X contains all of the training inputs, This should be a matrix of
            values, where x.shape[0] = n, where n is the number of
            available training points.
        y: numpy.array
            y contains the output target variables for each corresponding
            input vector. Again we expect y.shape[0] = n.
        """

        n, m = x.shape

        '''
        Prepare the namefile and the data, then write both to disk,
        then invoke cubist to train a regression tree
        '''

        # Prepare and write the namefile expected by cubist
        # TODO replace continuous with discrete for discrete data
        if self.feature_type is None:
            self.feature_type = np.zeros(m)

        d = {0: 'continuous', 1: 'discrete {}'.format(self.max_categories)}
        types = [d[k] for k in self.feature_type]

        names = ['t'] \
            + ['f{}: {}.'.format(j, t) for j, t in enumerate(types)]\
            + ['t: continuous.']
        namefile_string = '\n'.join(names)
        save_data(self._filename + '.names', namefile_string)

        # Write the data as a csv file for cubist's training
        y_copy = deepcopy(y)
        y_copy.shape = (n, 1)
        data = np.concatenate((x, y_copy), axis=1)
        np.savetxt(self._filename + '.data', data, delimiter=', ')

        # Run cubist and train the models
        self._run_cubist()

        '''
        Turn the model into a rule list, which we can evaluate
        '''

        # Get and store the output model and the required data
        modelfile = read_data(self._filename + '.model')

        # Split the modelfile into an array of models, where each model
        # contains some number of rules, hence the format:
        #   [[<Rule>, <Rule>, ...], [<Rule>, <Rule>, ...], ... ]
        models = map(remove_first_line, modelfile.split('rules')[1:])
        rules_split = [model.split('conds')[1:] for model in models]
        self._models = [list(map(Rule, model)) for model in rules_split]

        '''
        Complete the training by cleaning up after ourselves
        '''

        # Mark that we are now trained
        self._trained = True

        # Delete the files used during training
        self._remove_files(
            ['.tmp', '.names', '.data', '.model']
        )

    def predict_proba(self, x, interval=0.95):
        """ Predict the outputs and variances of the inputs
        This method predicts the output values that would correspond to
        each input in X. This method also returns the certainty of the
        model in each case, which is only sensible when the number of
        commitee members is greater than one.

        This method also outputs quantile information along with the
        variance to establish the probability distribution clearly.

        Parameters
        ----------
        x: numpy.array
            The inputs for which the model should be evaluated
        interval: float
            The probability threshold for which the quantiles should
            be output.

        Returns
        -------
        y_mean: numpy.array
            An array of expected output values given the inputs
        y_var: numpy.array
            The variance of the outputs
        ql: numpy.array
            The lower quantiles for each input
        qu: numpy.array
            The upper quantiles for each input
        """

        n, m = x.shape

        # We can't make predictions until we have trained the model
        if not self._trained:
            print('Train first')
            return

        # Run the rule list on each row of x one-by-one, incrementally
        #  calculating mean and variance
        y_mean = np.zeros(n)
        y_var = np.zeros(n)
        for j, model in enumerate(self._models):
            # FIXME: Can we vectorise this operation over n? It is quite slow
            for i, row in enumerate(x):
                for rule in model:
                    if rule.satisfied(row):
                        row_prediction = rule.regress(row)
                        delta = row_prediction - y_mean[i]
                        y_mean[i] += delta / (j + 1)
                        y_var[i] += delta * (row_prediction - y_mean[i])
                        break

        y_var /= len(self._models)

        # Determine quantiles
        ql, qu = norm.interval(interval, loc=y_mean, scale=np.sqrt(y_var))

        # Convert the prediction to a numpy array and return it
        return y_mean, y_var, ql, qu

    def predict(self, x):
        """ Predicts the y values that correspond to each input
        Just like predict_proba, this predicts the output value, given a
        list of inputs contained in x.

        Parameters
        ----------
        x: numpy.array
            The inputs for which the model should be evaluated

        Returns
        -------
        y_mean: numpy.array
            An array of expected output values given the inputs
        """

        mean, _, _, _ = self.predict_proba(x)
        return mean

    def _run_cubist(self):

        try:
            from uncoverml.cubist_config import invocation

        except ImportError:
            self._remove_files(['.tmp', '.names', '.data', '.model'])
            print('\nCubist not installed, please run makecubist first')
            import sys
            sys.exit()

        # Start the script and wait until it has yeilded
        command = (invocation +
                   (' -u' if self.unbiased else '') +
                   (' -r ' + str(self.max_rules)
                    if self.max_rules else '') +
                   (' -C ' + str(self.committee_members)
                    if self.committee_members else '') +
                   (' -f ' + self._filename))
        results = check_output(command, shell=True)
        # Print the program output directly
        if self.print_output:
            print(results.decode())

    def _remove_files(self, extensions):
        for extension in extensions:
            if os.path.exists(self._filename + extension):
                os.remove(self._filename + extension)


class Rule:

    comparator = {
        "<": operator.lt,
        ">": operator.gt,
        "=": operator.eq,
        ">=": operator.ge,
        "<=": operator.le
    }

    def __init__(self, rule):

        # Split the parts of the string so that they can be manipulated
        header, *conditions, polynomial = rule.split('\n')[:-1]

        '''
        Compute and store the regression variables
        '''
        # Split and parse the coefficients into variable/row indices,
        # coefficients and a bias unit for the regression
        bias, *splits = arguments(polynomial)
        variables, coefficients = (zip(*pairwise(splits))
                                   if len(splits)
                                   else ([], []))

        # Convert the regression values to numbers and store them
        self.bias = float(bias)
        self.variables = np.array([v[1:] for v in variables], dtype=int)
        self.coefficients = np.array(coefficients, dtype=float)

        '''
        Compute and store the condition evaluation variables
        '''

        self.conditions = [

            dict(type=CONTINUOUS,
                 operator=condition[3],
                 operand_index=int(condition[1][1:]),
                 operand_b=float(condition[2]))

            if int(condition[0]) == CONTINUOUS else

            dict(type=CATEGORICAL,
                 operand_index=int(condition[1][1:]),
                 values=parse_float_array(condition[2]))

            for condition in
            map(arguments, conditions)
        ]

    def satisfied(self, row):

        # Test that all of the conditions pass
        for condition in self.conditions:

            if condition['type'] == CONTINUOUS:
                comparison = self.comparator[condition['operator']]
                operand_a = row[condition['operand_index']]
                operand_b = condition['operand_b']
                if not comparison(operand_a, operand_b):
                    return False

            elif condition['type'] == CATEGORICAL:

                test_value = row[condition['operand_index']]
                allowed = condition['values']
                if not np.isclose(allowed, test_value, rtol=0.0001).any():
                    return False

        # If none of the conditions failed, the rule is satisfied
        return True

    def regress(self, row):

        prediction = self.bias + row[self.variables].dot(self.coefficients)
        return prediction
