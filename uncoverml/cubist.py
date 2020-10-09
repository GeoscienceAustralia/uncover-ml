# coding: utf-8
import os
from os.path import join, abspath
import pickle
import time
import random
import glob
from copy import deepcopy
from subprocess import check_output
from shlex import split as parse
import logging
import numpy as np
from scipy.stats import norm
import re
from collections import OrderedDict
import operator
import csv
from uncoverml import mpiops

_logger = logging.getLogger(__name__)
CONTINUOUS = 2
CATEGORICAL = 3
STR1 = re.compile('^Evaluation on training data', re.MULTILINE)
STR2 = re.compile('Evaluation on test data')
CASES = re.compile('cases\):\n')


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


def write_dict(filename, dict_obj):
    with open(filename, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dict_obj.items():
            writer.writerow([key, value])


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
                 max_rules=None, committee_members=1, max_categories=5000,
                 sampling=None, seed=None, neighbors=None, feature_type=None,
                 composite_model=False, auto=False, extrapolation=None,
                 calc_usage=False, bootstrap=None):
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
            whenever possible. If committee_members=1, committee model is not
            used
        max_categories: int
            The maximum number of categories cubist will search for in the
            data when creating a categorical variable.
        neighbors: int
            Number of  nearest–neighbors to adjust the predictions from
            the rule–based model. This option uses a composite model
            by combining it with an instance-based or nearest-neighbor model.
        sampling: float (0.1 - 99.9)
            percentage of data selected randomly by cubist
        seed: int
            random sampling seed
        feature_type:  numpy array
            An array of length equal to the number of features, 0 if
            that feature is continuous and 1 if it is categorical.
        extrapolation: float between 0-100
            allowed max deviation of predictions from training set
            targets range
        composite_model: bool
            whether to use composite model. False: used rule based model.
        auto: bool
            allow cubist to decide whether to use rule based or composite model
        """

        # Setting up the user details
        self._trained = False
        self.models = []
        self._filename = name + '_' + str(mpiops.rank_world) + '_' + \
            str(time.time()) + str(random.random())
        # Setting the user options
        self.print_output = print_output
        self.committee_members = committee_members
        self.unbiased = unbiased
        self.max_rules = max_rules
        self.feature_type = feature_type
        self.max_categories = max_categories
        self.neighbors = neighbors
        self.sampling = sampling
        self.auto = auto
        self.composite_model = composite_model
        self.calc_usage = calc_usage
        self.bootstrap = bootstrap

        if auto and composite_model:
            self.auto = False
            _logger.info('Both auto and composite model ware chosen. Disabling '
                     'auto and using composite model instead. To let cubist'
                     'auto decide, use composite_model=False, or comment out.')

        if composite_model and neighbors:
            _logger.info('Supplied neighbors will be used for composite model. '
                     'To let cubist decide the number of neighbors, do not '
                     'supply neighbors with config and choose '
                     'composite_model=True')
            self.composite_model = False

        if auto and neighbors:
            _logger.info('Supplied neighbors will be used for composite model. '
                     'To let cubist decide the number of neighbors, do not '
                     'supply neighbors with config.')
            self.auto = False

        # make sure seed is only used with sampling
        if (not sampling) and seed:
            seed = None
            _logger.info('Supplied random seed was not used as sampling % was not'
                     'provided')
        self.seed = seed
        self.extrapolation = extrapolation

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

        '''
        Prepare the namefile and the data, then write both to disk,
        then invoke cubist to train a regression tree
        '''
        n, m = x.shape

        # Prepare and write the namefile expected by cubist
        # TODO replace continuous with discrete for discrete data
        if self.feature_type is None:
            self.feature_type = {'f{}.tif'.format(v): 0 for v in range(m)}

        d = {0: 'continuous', 1: 'discrete {}'.format(self.max_categories)}
        types = OrderedDict()

        for k, v in self.feature_type.items():
            types[k] = d[v]

        names = ['t'] \
            + ['{}_{}: {}.'.format(k, i, v) for i, (k, v) in
               enumerate(types.items())]\
            + ['t: continuous.']
        namefile_string = '\n'.join(names)
        save_data(self._filename + '.names', namefile_string)

        # bootstrap
        if self.bootstrap:
            chosen = np.random.choice(range(len(y)),
                                      size=int(self.bootstrap/100.*len(y)),
                                      replace=True)
            y = y[chosen]
            x = x[chosen, :]

        n, _ = x.shape

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
            ['.tmp', '.names', '.data', '.model', '.pred']
        )

    def predict_dist(self, x, interval=0.95):
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
            _logger.warning(':mpi:Train first')
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
        Just like predict_dist, this predicts the output value, given a
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

        mean, _, _, _ = self.predict_dist(x)
        return mean

    def _run_cubist(self):

        try:
            from uncoverml.cubist_config import invocation

        except ImportError:
            self._remove_files(['.tmp', '.names', '.data', '.model'])
            _logger.error(':mpi:Cubist not installed, please run makecubist first')
            import sys
            sys.exit()

        # Start the script and wait until it has yielded
        command = (invocation +
                   (' -u' if self.unbiased else '') +
                   (' -r ' + str(self.max_rules)
                    if self.max_rules else '') +
                   (' -C ' + str(self.committee_members)
                    if self.committee_members > 1 else '') +
                   (' -n ' + str(self.neighbors)
                    if self.neighbors else '') +
                   (' -S ' + str(self.sampling)
                    if self.sampling else '') +
                   (' -e ' + str(self.extrapolation)
                    if self.extrapolation else '') +
                   (' -I ' + str(self.seed)
                    if self.seed else '') +
                   (' -i'
                    if self.composite_model else '') +
                   (' -a'
                    if self.auto else '') +
                   (' -f ' + self._filename))

        results = check_output(command, shell=True).decode()

        # Print the program output directly
        if self.print_output:
            print(results)

        if self.calc_usage:
            matched_str = CASES.split(STR1.split(results)[-1])[1]
            matched_str = STR2.split(matched_str)[0]
            save_data(self._filename + '.usg', matched_str)

    def _remove_files(self, extensions):
        for extension in extensions:
            if os.path.exists(self._filename + extension):
                os.remove(self._filename + extension)


class MultiCubist:
    """
    This is a wrapper on Cubist.
    """

    def __init__(self, outdir='.', trees=10, print_output=False, unbiased=True,
                 max_rules=None, committee_members=1, max_categories=5000,
                 neighbors=None, feature_type=None,
                 sampling=70, seed=None, extrapolation=None,
                 composite_model=False, auto=False, parallel=False,
                 calc_usage=False, bootstrap=None):
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

        self._trained = False

        # Setting the user options
        self.temp_dir = join(abspath(outdir), 'results')
        os.makedirs(self.temp_dir, exist_ok=True)
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
        self.extrapolation = extrapolation
        self.composite_model = composite_model
        self.auto = auto
        self.calc_usage = calc_usage
        self.bootstrap = bootstrap

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
        if 'NCPUS' in os.environ and 'PBS_NCPUS' in os.environ:
            num_nodes = int(int(os.getenv('PBS_NCPUS'))/int(os.getenv('NCPUS')))
            if num_nodes > 1:
                raise EnvironmentError(
                    "Can not use more than one node during MultiCubist "
                    "training.")

        # set a different random seed for each thread
        np.random.seed(mpiops.rank_world + np.random.randint(0, 10000))

        if self.parallel:  # during training
            process_trees = np.array_split(range(self.trees),
                                           mpiops.size_world)[mpiops.rank_world]
            temp_ = 'temp'
            temp_calc_usage = self.calc_usage
        else:  # during x val
            process_trees = range(self.trees)
            temp_ = 'temp_x_{}'.format(mpiops.rank_world)
            temp_calc_usage = False  # dont calc usage stats for x-val

        for t in process_trees:
            _logger.info(':mpi:training tree {} using process {}'.format(t, mpiops.rank_world))

            cube = Cubist(name=join(self.temp_dir, temp_ + '_{}'.format(t)),
                          print_output=self.print_output,
                          unbiased=self.unbiased,
                          max_rules=self.max_rules,
                          committee_members=self.committee_members,
                          max_categories=self.max_categories,
                          neighbors=self.neighbors,
                          feature_type=self.feature_type,
                          sampling=self.sampling,
                          extrapolation=self.extrapolation,
                          auto=self.auto,
                          composite_model=self.composite_model,
                          seed=np.random.randint(0, 10000),
                          calc_usage=temp_calc_usage,
                          bootstrap=self.bootstrap)
            cube.fit(x, y)
            if self.parallel:  # used in training
                pk_f = join(self.temp_dir,
                            'cube_{}.pk'.format(t))
            else:  # used when parallel is false, i.e., during x-val
                pk_f = join(self.temp_dir,
                            'cube_x_{}_p_{}.pk'.format(t, mpiops.rank_world))
            with open(pk_f, 'wb') as fp:
                pickle.dump(cube, fp)

        if self.parallel:
            mpiops.comm_world.barrier()
            # calc final usage stats
            if self.calc_usage and mpiops.rank_world == 0:
                self.calculate_usage()

        # Mark that we are now trained
        self._trained = True

    def predict_dist(self, x, interval=0.95):
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
            _logger.warning(':mpi:Train first')
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
                            'cube_x_{}_p_{}.pk'.format(i, mpiops.rank_world))
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
        Just like predict_dist, this predicts the output value, given a
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
        mean, _, _, _ = self.predict_dist(x)
        return mean

    def calculate_usage(self):
        """
        Averages the Cond and Model statistics of all the cubist runs

        Returns
        -------

        """
        results = OrderedDict()
        for t in range(self.trees):
            match = 'temp' + '_{}_'.format(t)

            # all usg files in results dir
            usgs = glob.glob(join(self.temp_dir, '*.usg'))
            names = [f for f in usgs if match in f]
            # sort by creation time
            names.sort(key=os.path.getctime, reverse=True)
            covariates = []
            with (open(names[0], 'r')) as f:  # use the latest usg
                for n, l in enumerate(f.readlines()):
                    l = l.rstrip()
                    if n > 8 and l:  # ignores empty lines too
                        args = [a.strip().replace('%', '')
                                for a in [l[:7], l[7:14], l[14:]]]
                        try:
                            cube_row = CubistReportRow(*args)
                        except TypeError:
                            _logger.error(":mpi:Parsing error. Cubist summary won' be created: {}"
                                          .format(args))


                        covariates.append(cube_row)
            results[t] = sorted(covariates, key=operator.attrgetter('feature'))

        ranking_cond = OrderedDict()
        ranking_model = OrderedDict()

        for i, f in enumerate(self.feature_type):
            ranking_cond[f.split('.tif')[0]] = []
            ranking_model[f.split('.tif')[0]] = []
        for t in range(self.trees):
            for k, c in results.items():
                for cc in c:
                    ranking_cond[
                        cc.feature.split('.tif_')[0]].append(
                        int(cc.cond) if cc.cond else 0)
                    ranking_model[cc.feature.split('.tif_')[0]].append(
                        int(cc.model) if cc.model else 0)

        for k, v in ranking_cond.items():
            if len(v):
                ranking_cond[k] = np.nanmean(v)
            else:
                ranking_cond[k] = 0.0

        for k, v in ranking_model.items():
            if len(v):
                ranking_model[k] = np.nanmean(v)
            else:
                ranking_cond[k] = 0

        write_dict(join(self.temp_dir, 'ranking_cond.csv'), ranking_cond)
        write_dict(join(self.temp_dir, 'ranking_model.csv'), ranking_model)


class CubistReportRow:
    """
    convenience class for accumulating cubist report
    """
    def __init__(self, cond, model, feature):
        self.cond = cond
        self.model = model
        self.feature = feature


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

        # py27 compat
        # rule_splits = rule.split('\n')[:-1]
        # # header = rule_splits[0]
        # conditions = rule_splits[1:-1]
        # polynomial = rule_splits[-1]
        '''
        Compute and store the regression variables
        '''

        # Split and parse the coefficients into variable/row indices,
        # coefficients and a bias unit for the regression
        bias, *splits = arguments(polynomial)
        # py27 compat
        # args = arguments(polynomial)
        # bias, splits = args[0], args[1:]

        v, c = (zip(*pairwise(splits))
                if len(splits)
                else ([], []))

        # Convert the regression values to a coefficient vector
        self.bias = float(bias)
        variables = np.array([v.split('.tif_')[-1] for v in v], dtype=int)
        coefficients = np.array(c, dtype=float)
        self.coefficients = np.zeros(m)
        self.coefficients[variables] = coefficients

        '''
        Compute and store the condition evaluation variables
        '''

        self.conditions = [

            dict(type=CONTINUOUS,
                 operator=condition[3],
                 operand_index=int(condition[1].split('.tif_')[-1]),
                 operand=float(condition[2]))

            if int(condition[0]) == CONTINUOUS else

            dict(type=CATEGORICAL,
                 operand_index=int(condition[1].split('.tif_')[-1]),
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
