"""
A simple implementation of DBSCAN for 1D data.

It should be *much* more efficient for large datasets.
"""
from typing import Optional

import numpy as np


class DBSCAN1D:
    """
    A one dimensional implementation of DBSCAN.

    This class has a very similar interface as sklearn's implementation. In
    most cases they should be interchangeable.
    """

    # params that change upon fit/training
    core_sample_indices_: Optional[np.ndarray] = None
    components_: Optional[np.ndarray] = None
    labels_: Optional[np.ndarray] = None

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples

    def _get_is_core(self, ar):
        """ Determine if each point is a core. """
        mineps = np.searchsorted(ar, ar - self.eps, side="left")
        maxeps = np.searchsorted(ar, ar + self.eps, side="right")
        core = (maxeps - mineps) >= self.min_samples
        return core

    def _assign_core_group_numbers(self, cores):
        """ Given a group of core points, assign group numbers to each. """
        gt_eps = abs(cores - np.roll(cores, 1)) > self.eps
        # The first value doesn't need to be compared to last, set to False so
        # that cluster names are consistent (see issue #3).
        if len(gt_eps):
            gt_eps[0] = False
        return gt_eps.astype(int).cumsum()

    def _bound_on(self, arr, max_len):
        """ Ensure all values in array are bounded between 0 and max_len. """
        arr[arr < 0] = 0
        arr[arr >= max_len] = max_len - 1
        return arr

    def _get_non_core_labels(self, non_cores, cores, core_nums):
        """ Get labels for non-core points. """
        # start out with noise labels (-1)
        out = (np.ones(len(non_cores)) * -1).astype(int)
        if not len(cores):  # there are no core points, bail out early
            return out
        # get index where non-core point would be inserted into core points
        cc_right = np.searchsorted(cores, non_cores)
        cc_left = cc_right - 1
        # make sure these respect bounds of cores
        cc_left = self._bound_on(cc_left, len(cores))
        cc_right = self._bound_on(cc_right, len(cores))
        # now get index and values of closest core point (on right and left)
        index = np.array([cc_right, cc_left]).T
        vals = np.array([cores[cc_right], cores[cc_left]]).T
        # calculate the difference between each non-core and its neighbor cores
        diffs = abs(vals - np.array([non_cores, non_cores]).T)
        argmin = diffs.argmin(axis=1)
        dummy_ = np.arange(0, len(diffs))
        min_vals = diffs[dummy_, argmin]
        inds = index[dummy_, argmin]
        # determine if closest core point is close enough to assign to group
        is_connected = min_vals <= self.eps
        # update group and return
        out[is_connected] = core_nums[inds[is_connected]]
        return out

    def fit(self, X, y=None, sample_weight=None):
        """
        Performing DBSCAN clustering on 1D array.

        Parameters
        ----------
        X
            The input array
        y
            Not used
        sample_weight
            Not yet supported
        """
        assert len(X.shape) == 1 or X.shape[-1] == 1, "X must be 1d array"
        assert y is None, "y parameter is ignored"
        assert sample_weight is None, "sample weights are not yet supported"
        # get sorted array and sorted order
        ar = X.flatten()
        ar_sorted = np.sort(ar)
        undo_sorted = np.argsort(np.argsort(ar))

        # get core points, and separate core from non-core
        is_core = self._get_is_core(ar_sorted)
        group_nums = np.ones_like(is_core) * -1  # empty group numbers
        cores = ar_sorted[is_core]
        non_cores = ar_sorted[~is_core]
        # get core numbers and non-core numbers
        core_nums = self._assign_core_group_numbers(cores)
        non_core_nums = self._get_non_core_labels(non_cores, cores, core_nums)
        group_nums[is_core] = core_nums
        group_nums[~is_core] = non_core_nums
        # unsort group nums and core indices
        out = group_nums[undo_sorted]
        is_core_original_sorting = is_core[undo_sorted]
        # set class attrs and return predicted labels
        self.core_sample_indices_ = np.where(is_core_original_sorting)
        # self.components_ = cores.values
        self.labels_ = out
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        """
        Performing DBSCAN clustering on 1D array and return the label array.

        Parameters
        ----------
        X
            The input array
        y
            Not used
        sample_weight
            Not yet supported
        """
        self.fit(X, y=y, sample_weight=sample_weight)
        return self.labels_
