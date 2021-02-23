# DBSCAN1D

[![Coverage](https://codecov.io/gh/d-chambers/dbscan1d/branch/master/graph/badge.svg)](https://codecov.io/gh/d-chambers/dbscan1d)
[![Supported Versions](https://img.shields.io/pypi/pyversions/dbscan1d.svg)](https://pypi.python.org/pypi/dbscan1d)
[![PyPI](https://pepy.tech/badge/dbscan1d)](https://pepy.tech/project/dbscan1d)
[![Licence](https://www.gnu.org/graphics/lgplv3-88x31.png)](https://www.gnu.org/licenses/lgpl.html)

dbscan1d is a 1D implementation of the [DBSCAN algorithm](https://en.wikipedia.org/wiki/DBSCAN). It was created to efficiently
preform clustering on large 1D arrays.

[Sci-kit Learn's DBSCAN implementation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) does
not have a special case for 1D, where calculating the full distance matrix is wasteful. It is much better to simply sort
the input array and performing efficient bisects for finding closest points. Here are the results of running the simple
profile script included with the package. In every case DBSCAN1D is much faster than scikit learn's implementation.

![image](https://github.com/d-chambers/dbscan1d/raw/master/profile_results.png)

## Installation
Simply use pip to install dbscan1d:
```bash
pip install dbscan1d
```
It only requires numpy.

## Quickstart
dbscan1d is designed to be interchangable with sklearn's implementation in almost
all cases. The exception is that the `weights` parameter is not yet supported.

```python
from sklearn.datasets import make_blobs

from dbscan1d.core import DBSCAN1D

# make blobs to test clustering
X = make_blobs(1_000_000, centers=2, n_features=1)[0]

# init dbscan object
dbs = DBSCAN1D(eps=.5, min_samples=4)

# get labels for each point
labels = dbs.fit_predict(X)

# show core point indices
dbs.core_sample_indices_

# get values of core points
dbs.components_
```
