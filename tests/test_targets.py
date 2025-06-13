import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from uncoverml import targets


@pytest.fixture
def sample_data():
    positions = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    observations = np.array([10.0, 20.0, 30.0])
    groups = np.array([0, 1, 1])
    weights = np.array([1.0, 0.5, 0.2])
    othervals = {'extra': np.array([100, 200, 300])}
    keep = np.array([True, False, True])
    targets_obj = targets.Targets(positions, observations, groups, weights, othervals)
    return targets_obj, keep, positions, observations, groups, weights, othervals


@patch('uncoverml.targets.mpiops.comm')
def test_gather_targets_main(mock_comm, sample_data):
    targets_obj, keep, positions, observations, groups, weights, othervals = sample_data
    mock_comm.allgather.side_effect = lambda x: [x]
    result = targets.gather_targets_main(targets_obj, keep, node=None)
    np.testing.assert_array_equal(result.positions, positions[keep])
    np.testing.assert_array_equal(result.observations, observations[keep])
    np.testing.assert_array_equal(result.groups, groups[keep])
    np.testing.assert_array_equal(result.weights, weights[keep])
    np.testing.assert_array_equal(result.fields['extra'], othervals['extra'][keep])


@patch('uncoverml.targets.mpiops.comm')
def test_gather_targets_main_node(mock_comm, sample_data):
    targets_obj, keep, positions, observations, groups, weights, othervals = sample_data
    mock_comm.gather.side_effect = lambda x, root: [x]
    result = targets.gather_targets_main(targets_obj, keep, node=1)
    np.testing.assert_array_equal(result.positions, positions[keep])
    np.testing.assert_array_equal(result.observations, observations[keep])
    np.testing.assert_array_equal(result.groups, groups[keep])
    np.testing.assert_array_equal(result.weights, weights[keep])


@patch('uncoverml.targets.gather_targets_main')
def test_gather_targets(mock_main, sample_data):
    targets_obj, keep, *_ = sample_data
    mock_main.return_value = 'mock_result'
    result = targets.gather_targets(targets_obj, keep, config=None, node=0)
    mock_main.assert_called_once_with(targets_obj, keep, 0)
    assert result == 'mock_result'


@patch('numpy.savetxt')
def test_save_dropped_targets(mock_savetxt, sample_data):
    targets_obj, keep, *_ = sample_data
    mock_config = MagicMock()
    mock_config.output_dir = '/tmp'
    targets.save_dropped_targets(mock_config, keep, targets_obj)
    mock_savetxt.assert_called_once()
