import os
import random
import time
from os.path import join
import pickle
from copy import deepcopy
from subprocess import check_output
from shlex import split as parse
import logging
import numpy as np
from scipy.stats import norm

from uncoverml import mpiops

log = logging.getLogger(__name__)
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
                 sampling=60.0, seed=1,
                 neighbors=5, feature_type=None):
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
        neighbors: int
            Number of  nearest–neighbors to adjust the predictions from
            the rule–based model.
        sampling: float (0.1 - 99.9)
            percentage of data selected randomly by cubist
        seed: int
            random sampling seed
        feature_type:  numpy array
            An array of length equal to the number of features, 0 if
            that feature is continuous and 1 if it is categorical.
        """

        # Setting up the user details
        self._trained = False
        self.models = []
        self._filename = name + str(time.time()) + str(random.random())

        # Setting the user options
        self.print_output = print_output
        self.committee_members = committee_members
        self.unbiased = unbiased
        self.max_rules = max_rules
        self.feature_type = feature_type
        self.max_categories = max_categories
        self.neighbors = neighbors
        self.sampling = sampling
        self.seed = seed

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

        # Define a function that assigns the number of rows to the rule
        def new_rule(model):
            return Rule(model, m)

        # Split the modelfile into an array of models, where each model
        # contains some number of rules, hence the format:
        #   [[<Rule>, <Rule>, ...], [<Rule>, <Rule>, ...], ... ]
        models = map(remove_first_line, modelfile.split('rules')[1:])
        rules_split = [model.split('conds')[1:] for model in models]
        self.models = [list(map(new_rule, model)) for model in rules_split]

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

        # Determine which rule to run on each row and then run the regression
        # on each row of x to get the regression output.
        y_pred = np.zeros((n, len(self.models)))
        for m, model in enumerate(self.models):
            for rule in model:

                # Determine which rows satisfy this rule
                mask = rule.satisfied(x)

                # Make the prediction for the whole matrix, and keep only the
                # rows that are correctly sized
                y_pred[mask, m] += rule.regress(x, mask)

        y_mean = np.mean(y_pred, axis=1)
        y_var = np.var(y_pred, axis=1)

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
                   (' -n ' + str(self.neighbors)
                    if self.neighbors else '') +
                   (' -S ' + str(self.sampling)
                    if self.sampling else '') +
                   (' -I ' + str(self.seed)
                    if self.seed else '') +
                   (' -f ' + self._filename))
        results = check_output(command, shell=True)
        # Print the program output directly
        if self.print_output:
            print(results.decode())

    def _remove_files(self, extensions):
        for extension in extensions:
            if os.path.exists(self._filename + extension):
                os.remove(self._filename + extension)


def _join_dicts(dicts):
    if dicts is None:
        return
    d = {k: v for D in dicts for k, v in D.items()}
    return d


class MultiCubist:
    """
    This is a wrapper on Cubist.
    """

    def __init__(self, outdir='.', trees=100, print_output=False, unbiased=True,
                 max_rules=None, committee_members=20, max_categories=5000,
                 neighbors=5, feature_type=None,
                 sampling=60, seed=1,
                 parallel=False):
        """
        Instantiate the multicubist class with a number of invocation
        parameters

        Parameters
        ----------
        trees: int
            number of Cubist trees
        parallel: bool
            Whether to use mpi for fitting or not

        Other Parameters definitions can be found in Cubist.
        """

        # Setting up the user details
        self._trained = False

        # Setting the user options
        self.temp_dir = outdir
        self.print_output = print_output
        self.committee_members = committee_members
        self.unbiased = unbiased
        self.max_rules = max_rules
        self.feature_type = feature_type
        self.max_categories = max_categories
        self.neighbors = neighbors
        self.trees = trees
        self.seed = seed
        self.sampling = sampling
        self.parallel = parallel

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
        # set a different random seed for each thread
        np.random.seed(self.seed + mpiops.chunk_index)

        if self.parallel:
            process_trees = np.array_split(range(self.trees),
                                           mpiops.chunks)[mpiops.chunk_index]
        else:
            process_trees = range(self.trees)

        for t in process_trees:
            print('training tree {} using '
                  'process {}'.format(t, mpiops.chunk_index))

            cube = Cubist(name='temp_' + str(t) + '_',
                          print_output=self.print_output,
                          unbiased=self.unbiased,
                          max_rules=self.max_rules,
                          committee_members=self.committee_members,
                          max_categories=self.max_categories,
                          neighbors=self.neighbors,
                          feature_type=self.feature_type,
                          sampling=self.sampling,
                          seed=np.random.randint(0, 10000))
            cube.fit(x, y)
            if self.parallel:  # used in training
                pk_f = join(self.temp_dir,
                            'cube_{}.pk'.format(t))
            else:  # used when parallel is false, i.e., during x-val
                pk_f = join(self.temp_dir,
                            'cube_{}_{}.pk'.format(t, mpiops.chunk_index))
            with open(pk_f, 'wb') as fp:
                pickle.dump(cube, fp)

        if self.parallel:
            mpiops.comm.barrier()

        # Mark that we are now trained
        self._trained = True

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

        # We can't make predictions until we have trained the model
        if not self._trained:
            print('Train first')
            return

        n, _ = x.shape

        # on each row of x to get the regression output.
        # we have prediction for each x tree/cubes * len(models) in each tree

        y_pred = np.zeros((n, self.trees * self.committee_members))

        for i in range(self.trees):
            if self.parallel:  # used in training
                pk_f = join(self.temp_dir,
                            'cube_{}.pk'.format(i))
            else:  # used when parallel is false, i.e., during x-val
                pk_f = join(self.temp_dir,
                            'cube_{}_{}.pk'.format(i, mpiops.chunk_index))
            with open(pk_f, 'rb') as fp:
                c = pickle.load(fp)
                for m, model in enumerate(c.models):
                    for rule in model:
                        # Determine which rows satisfy this rule
                        mask = rule.satisfied(x)
                        # Make the prediction for the whole matrix,
                        # and keep only the rows that are correctly sized
                        y_pred[mask, i * self.committee_members + m] += \
                            rule.regress(x, mask)

        y_mean = np.mean(y_pred, axis=1)
        y_var = np.var(y_pred, axis=1)

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


class Rule:

    comparator = {
        "<": np.less,
        ">": np.greater,
        "=": np.equal,
        ">=": np.greater_equal,
        "<=": np.less_equal
    }

    def __init__(self, rule, m):

        # Split the parts of the string so that they can be manipulated
        header, *conditions, polynomial = rule.split('\n')[:-1]

        '''
        Compute and store the regression variables
        '''

        # Split and parse the coefficients into variable/row indices,
        # coefficients and a bias unit for the regression
        bias, *splits = arguments(polynomial)
        v, c = (zip(*pairwise(splits))
                if len(splits)
                else ([], []))

        # Convert the regression values to a coefficient vector
        self.bias = float(bias)
        variables = np.array([v[1:] for v in v], dtype=int)
        coefficients = np.array(c, dtype=float)
        self.coefficients = np.zeros(m)
        self.coefficients[variables] = coefficients

        '''
        Compute and store the condition evaluation variables
        '''

        self.conditions = [

            dict(type=CONTINUOUS,
                 operator=condition[3],
                 operand_index=int(condition[1][1:]),
                 operand=float(condition[2]))

            if int(condition[0]) == CONTINUOUS else

            dict(type=CATEGORICAL,
                 operand_index=int(condition[1][1:]),
                 values=parse_float_array(condition[2]))

            for condition in
            map(arguments, conditions)
        ]

    def satisfied(self, x):

        # Define a mask for each row in x
        mask = np.ones(len(x), dtype=bool)

        # Test that all of the conditions pass
        for condition in self.conditions:

            if condition['type'] == CONTINUOUS:
                comparison = self.comparator[condition['operator']]
                x_column = x[:, condition['operand_index']]
                operand = condition['operand']
                mask &= comparison(x_column, operand)

            elif condition['type'] == CATEGORICAL:
                allowed = condition['values']
                x_column = x[:, [condition['operand_index']]]
                mask &= np.isclose(allowed, x_column).any(axis=1)

        # If all of the conditions passed for a single row, we can conclude
        # that this row satisfies this rule
        return mask

    def regress(self, x, mask=None):

        prediction = self.bias + x[mask].dot(self.coefficients)
        return prediction
