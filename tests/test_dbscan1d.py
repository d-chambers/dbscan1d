"""
Tests for dbscan1d.

Requires sklearn.
"""
import copy
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

import dbscan1d
from dbscan1d.core import DBSCAN1D

TEST_DATA_PATH = Path(__file__).parent / "test_data"


def _bound_on(arr, max_len):
    """Ensure all values in array are bounded between 0 and max_len."""
    arr[arr < 0] = 0
    arr[arr >= max_len] = max_len - 1
    return arr


def _check_points_between_cores(dbs_1, dbs_2, data):
    """
    Handle the corner case of points between two core points.

    The groups are different. Check if this is due to a point within
    eps of two groups. DBScan1D should have assigned it to the closest
    group. See the note in the readme for more info.
    """
    if not np.all(dbs_1.core_sample_indices_ == dbs_2.core_sample_indices_):
        return False

    # make life easier and sort input data
    data = np.sort(data[:, 0])
    dbs_copy = copy.deepcopy(dbs_1)
    dbs_copy.fit(data[:, np.newaxis])

    core_inds = dbs_copy.core_sample_indices_
    non_core_inds = np.setdiff1d(np.arange(len(data)), core_inds)
    labels = dbs_copy.labels_
    eps = dbs_copy.eps

    core_data = data[core_inds]
    non_core_data = data[non_core_inds]

    right_inds = np.searchsorted(core_data, non_core_data, side="left")
    right_inds = _bound_on(right_inds, len(dbs_1.core_sample_indices_))
    left_inds = _bound_on(right_inds - 1, len(dbs_1.core_sample_indices_))
    for ind, rind, lind in zip(non_core_inds, right_inds, left_inds):
        val = data[ind]
        right_diff = np.abs(data[core_inds[rind]] - val)
        left_diff = np.abs(data[core_inds[lind]] - val)
        # No right answer, difference are the same between both
        if left_diff == right_diff:
            continue
        # This point should have no group
        elif right_diff > eps and left_diff > eps and labels[ind] == -1:
            continue
        elif right_diff < left_diff and labels[ind] != labels[core_inds[rind]]:
            return False
        elif left_diff < right_diff and labels[ind] != labels[core_inds[lind]]:
            return False
    return True


def clusters_equivalent(dbs_1: DBSCAN1D, dbs_2: DBSCAN, data: np.ndarray):
    """
    Return True if two arrays defining cluster structures are equivalent.

    Accounts for each array having a different name for each cluster.

    Parameters
    ----------
    dbs_1
        The DBSCAN1D instance after training
    dbs_2
        The DBSCAN instance (from sklearn) after training
    data
        The input data for training.
    """
    labels1 = dbs_1.labels_
    labels2 = dbs_2.labels_
    array = np.array([labels1, labels2]).T
    # get the unique pairings of data
    unique_pairs = np.unique(array, axis=0)
    # Ensure each number is paired with exactly one other number
    left = np.unique(unique_pairs[:, 0])
    right = np.unique(unique_pairs[:, 1])
    # The groups aren't the same!
    if not len(left) == len(right) == len(unique_pairs):
        return _check_points_between_cores(dbs_1, dbs_2, data)
    return True


def get_unclustered(labels1, labels2):
    """
    Return two lists, first of points clustered in labels1 but not 2, then
    the opposite.
    """
    out1 = np.where((labels1 == -1) & (labels2 != -1))[0]
    out2 = np.where((labels1 != -1) & (labels2 == -1))[0]
    return out1, out2


def unclusterd_equal(labels1, labels2):
    """
    Return True if unclustered points are the same in both arrays.
    """
    un1, un2 = get_unclustered(labels1, labels2)
    return len(un1) == len(un2) == 0


def generate_test_data(num_points, centers=None):
    """Generate data for testing."""
    blobs, blob_labels = make_blobs(
        num_points, n_features=1, centers=centers, random_state=13
    )
    X = blobs.flatten()
    np.random.shuffle(X)
    return X, blob_labels


# --- tests cases


class TestSKleanEquivalent:
    """
    Basic tests for DBSCAN1D.

    Essentially these just ensure the output is equivalent to sklearn's dbscan
    for various contrived datasets.
    """

    # define a small range of dbscan input params over which tests will
    # be parametrized
    eps_values = [0.0001, 0.1, 0.5, 1, 2]
    min_samples_values = [1, 2, 5, 15]
    db_params = list(product(eps_values, min_samples_values))

    centers = [
        np.array([0, 5, 10]),
        np.arange(10),
        np.array([1, 2, 3, 4, 5, 10]),
        np.array([1, 1.1, 1.2, 1.3, 1.4, 1.5]),
        2,
        7,
    ]

    @pytest.fixture(scope="class", params=centers)
    def blobs(self, request):
        """The first sklearn-generated blobs."""
        centers = request.param
        if isinstance(centers, np.ndarray):
            centers = centers.reshape(1, -1)
        X, _ = generate_test_data(1000, centers=centers)
        return X.reshape(-1, 1)

    @pytest.fixture(scope="class", params=db_params)
    def db_instances(self, request):
        """
        Using the parametrized values, unit an instance of DBSCAN1D and
        DBSCAN (from sklearn)
        """
        eps, min_samples = request.param
        db1 = DBSCAN1D(eps=eps, min_samples=min_samples)
        db2 = DBSCAN(eps=eps, min_samples=min_samples)
        return db1, db2

    def test_blob1_outputs(self, blobs, db_instances):
        """The first tests case."""
        db1, db2 = db_instances
        out1 = db1.fit_predict(blobs)
        out2 = db2.fit_predict(blobs)

        # First make sure the same samples were flagged as cores
        assert np.equal(db1.core_sample_indices_, db2.core_sample_indices_).all()
        # Then make sure all points left unclustered on one array are not
        # clustered on the second.
        assert unclusterd_equal(out1, out2)
        # Now assert the same points fall in the same cluster groups
        assert clusters_equivalent(db1, db2, blobs)


class TestIssues:
    """Tests for issues filled on github."""

    @pytest.fixture(scope="class", params=(TEST_DATA_PATH.glob("issue_7_*.npy")))
    def issue_7_array(self, request):
        """Yield all the issue 7 arrays for testing."""
        ar = np.load(request.param)
        return ar

    def test_issue_3(self):
        """
        Test that cluster numbers remain consistent for 1 cluster. See issue #3.
        """
        ar1 = [86400.0, 86401.0, 86399.0, 86401.0, 86399.0, 86401.0]
        ar2 = [46823, 46818, 46816, 46816, 46819]
        dbscan = DBSCAN1D(eps=5, min_samples=3)
        # Group names should be sequential, starting with zero
        out1 = dbscan.fit_predict(np.array(ar2))
        out2 = dbscan.fit_predict(np.array(ar1))
        assert set(out1) == {0}
        assert set(out2) == {0}

    def test_issue_6(self):
        """
        DBSCAN1D should raise value error if anything but euclidean
        distance is selected.
        """
        with pytest.raises(ValueError, match="euclidean"):
            DBSCAN1D(metric="cosine")

    def test_version(self):
        """Test that a version number is returned."""
        version = dbscan1d.__version__
        assert version is not None and version != "0.0.0"

    def test_issue_7(self, issue_7_array):
        """Test that core points and labels are consistent."""
        dbs_1 = DBSCAN1D(eps=0.5, min_samples=4)
        dbs_2 = DBSCAN(eps=0.5, min_samples=4)
        # get labels for each point, assert they are equivalent
        dbs_1.fit_predict(issue_7_array)
        dbs_2.fit_predict(issue_7_array)
        assert clusters_equivalent(dbs_1, dbs_2, issue_7_array)
        # also test indices of core points
        assert np.all(dbs_1.core_sample_indices_ == dbs_2.core_sample_indices_)
