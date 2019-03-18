
import numpy as np
from sklearn.metrics import r2_score

from uncoverml.cubist import Cubist, MultiCubist

# Declare some test data taken from the boston houses dataset
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


def test_correct_range():

    # Fit the data
    predictor = Cubist(print_output=False,
                       sampling=90, seed=0, committee_members=2)
    predictor.fit(x, y)

    # Predict the output
    y_pred = predictor.predict(x)

    # Assert that the true y is similar to the prediction
    score = r2_score(y, y_pred)
    assert 0.68 < score < 0.8


def test_correct_range_with_sampling():

    # Fit the data
    predictor = Cubist(print_output=False,
                       sampling=90, seed=10, committee_members=2)
    predictor.fit(x, y)

    # Predict the output
    y_pred = predictor.predict(x)

    # Assert that the true y is similar to the prediction
    score = r2_score(y, y_pred)
    assert 0.68 < score < 0.73


def test_multicubist():
    predictor = MultiCubist(print_output=False,
                            trees=5,
                            sampling=90,
                            seed=1,
                            neighbors=1)
    predictor.fit(x, y)

    # Predict the output
    y_pred = predictor.predict(x)

    # Assert that the true y is similar to the prediction
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

    # Predict the output
    y_pred_p = predictor.predict(x)

    score = r2_score(y, y_pred_p)

    assert 0.5 < score < 0.8
