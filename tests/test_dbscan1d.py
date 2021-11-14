"""
Tests for dbscan1d.

Requires sklearn.
"""
from itertools import product

import numpy as np
import pytest
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs

from dbscan1d.core import DBSCAN1D


# --- tests utils


def clusters_equivalent(labels1, labels2):
    """
    Return True if two arrays defining cluster structures are equivalent.

    Accounts for each array having a different name for each cluster.
    """
    array = np.array([labels1, labels2]).T
    # get the unique pairings of data
    unique_pairs = np.unique(array, axis=0)
    # Ensure each number is paired with exactly one other number
    left = np.unique(unique_pairs[:, 0])
    right = np.unique(unique_pairs[:, 1])
    return len(left) == len(right) == len(unique_pairs)


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
    min_samples_values = [0, 1, 2, 5, 15]
    # eps_values = [ .5,]
    # min_samples_values = [ 2,]
    db_params = list(product(eps_values, min_samples_values))

    @pytest.fixture(scope="class")
    def blobs1(self):
        """ The first sklearn-generated blobs. """
        centers = np.array([0, 5, 10]).reshape(1, -1)
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

    def test_blob1_outputs(self, blobs1, db_instances):
        """ The first tests case. """
        db1, db2 = db_instances
        out1 = db1.fit_predict(blobs1)
        out2 = db2.fit_predict(blobs1)

        # First make sure the same samples were flagged as cores
        assert np.equal(db1.core_sample_indices_, db2.core_sample_indices_).all()
        # Then make sure all points left unclustered on one array are not
        # clustered on the second.
        assert unclusterd_equal(out1, out2)
        # Now assert the same points fall in the same cluster groups
        assert clusters_equivalent(out1, out2)


class TestIssues:
    """Tests for issues filled on github."""

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
        with pytest.raises(ValueError, match='euclidean'):
            DBSCAN1D(metric='cosine')
