import os
import tempfile
import csv
import numpy as np
from unittest.mock import patch, mock_open
from sklearn.metrics import r2_score
from uncoverml.cubist import (
    Cubist, MultiCubist, write_dict, mean,
    variance_with_mean, cond_line,parse_float_array
)

x = np.array([
        [0.006, 18.00, 2.310, 0.5380, 6.5750, 65.20, 4.0900, 1, 296.0, 15.30],
        [0.027, 0.00, 7.070, 0.4690, 6.4210, 78.90, 4.9671, 2, 242.0, 17.80],
        [0.027, 0.00, 7.070, 0.4690, 7.1850, 61.10, 4.9671, 2, 242.0, 17.80],
        [0.032, 0.00, 2.180, 0.4580, 6.9980, 45.80, 6.0622, 3, 222.0, 18.70],
        [0.069, 0.00, 2.180, 0.4580, 7.1470, 54.20, 6.0622, 3, 222.0, 18.70],
        [0.029, 0.00, 2.180, 0.4580, 6.4300, 58.70, 6.0622, 3, 222.0, 18.70],
        [0.088, 12.50, 7.870, 0.5240, 6.0120, 66.60, 5.5605, 5, 311.0, 15.20],
        [0.144, 12.50, 7.870, 0.5240, 6.1720, 96.10, 5.9505, 5, 311.0, 15.20],
        [0.211, 12.50, 7.870, 0.5240, 5.6310, 100.00, 6.0821, 5, 311.0, 15.20],
        [0.170, 12.50, 7.870, 0.5240, 6.0040, 85.90, 6.5921, 5, 311.0, 15.20],
        [0.224, 12.50, 7.870, 0.5240, 6.3770, 94.30, 6.3467, 5, 311.0, 15.20],
        [0.117, 12.50, 7.870, 0.5240, 6.0090, 82.90, 6.2267, 5, 311.0, 15.20],
        [0.093, 12.50, 7.870, 0.5240, 5.8890, 39.00, 5.4509, 5, 311.0, 15.20],
        [0.629, 0.00, 8.140, 0.5380, 5.9490, 61.80, 4.7075, 4, 307.0, 21.00],
        [0.637, 0.00, 8.140, 0.5380, 6.0960, 84.50, 4.4619, 4, 307.0, 21.00]])

y = np.array([24.00, 21.60, 34.70, 33.40, 36.20, 28.70, 22.90, 27.10,
                  16.50, 18.90, 15.00, 18.90, 21.70, 20.40, 18.2])


def test_cond_line(): 
    assert cond_line('conds') == 1
    assert cond_line('abc') == 0


@patch('builtins.open', new_callable=mock_open)
def test_write_dict(mock_file):
    input_dict = {'a': 1, 'b': 2}
    write_dict('mock.csv', input_dict)
    mock_file.assert_called_once_with('mock.csv', 'w')


def test_mean(): 
    numbers = {1,2,3,4}
    result = mean(numbers) 
    assert result == 2.5


def test_variance_with_mean():
    numbers = [1, 2, 3, 4]
    mean_value = sum(numbers) / len(numbers)
    var_func = variance_with_mean(mean_value)
    assert var_func(numbers) == 5.0


def test_parse_float_array():
    input_string = '1.0,2.0,3.0,4'
    expected = [1.0, 2.0, 3.0, 4.0]
    result = parse_float_array(input_string)
    assert result == expected


def test_correct_range():

    predictor = Cubist(print_output=False,
                       sampling=90, seed=0, committee_members=2)
    predictor.fit(x, y)
    y_pred = predictor.predict(x)
    score = r2_score(y, y_pred)
    assert 0.68 < score < 0.8


def test_correct_range_with_sampling():
    predictor = Cubist(print_output=False,
                       sampling=90, seed=10, committee_members=2)
    predictor.fit(x, y)
    y_pred = predictor.predict(x)
    score = r2_score(y, y_pred)
    assert 0.68 < score < 0.73


def test_multicubist():
    predictor = MultiCubist(print_output=False,
                            trees=5,
                            sampling=90,
                            seed=1,
                            neighbors=1)
    predictor.fit(x, y)
    y_pred = predictor.predict(x)
    score = r2_score(y, y_pred)
    assert 0.5 < score < 0.8


def test_multicibist_mpi(mpisync):
    """
    run this with something like:
    "mpirun -np 4 py.test ../tests/test_cubist.py::test_multicubist_mpi"

    """
    predictor = MultiCubist(trees=10,
                            sampling=60,
                            seed=1,
                            neighbors=1,
                            committee_members=5,
                            parallel=True)
    predictor.fit(x, y)
    y_pred_p = predictor.predict(x)
    score = r2_score(y, y_pred_p)
    assert 0.5 < score < 0.8

@patch('glob.glob')
@patch('os.path.getctime')
@patch('builtins.open', new_callable=mock_open, read_data="""


feat1.tif_0     10     20
feat2.tif_0     5      15
""")
@patch('uncoverml.cubist.write_dict')
def test_calculate_usage(mock_write, mock_open_fn, mock_ctime, mock_glob):
    temp_dir = tempfile.mkdtemp()
    feature_type = {'feat1.tif': 0, 'feat2.tif': 0}
    mock_usg_file = os.path.join(temp_dir, 'temp_0_0.usg')
    mock_glob.return_value = [mock_usg_file]
    mock_ctime.return_value = 0
    predictor = MultiCubist(outdir=temp_dir, trees=1, calc_usage=True)
    predictor.feature_type = feature_type
    predictor.temp_dir = temp_dir
    predictor.calculate_usage()
    assert mock_write.call_count == 2
