import numpy as np

from opera_utils.disp._reader import _get_border


def test_get_border():
    a = np.arange(24).reshape(2, 3, 4)
    # This is what `a` looks like:
    # In [11]: a = np.arange(24).reshape(2,3,4)
    # In [12]: a
    # Out[12]:
    # array([[[ 0,  1,  2,  3],
    #         [ 4,  5,  6,  7],
    #         [ 8,  9, 10, 11]],

    #        [[12, 13, 14, 15],
    #         [16, 17, 18, 19],
    #         [20, 21, 22, 23]]])
    assert np.allclose(_get_border(a), np.array([[[5.5]], [[17.5]]]))
