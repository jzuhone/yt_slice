from yt.frontends.stream.api import load_uniform_grid
from yt.mods import FixedResolutionBuffer
import numpy as np
import pytest


class YtSlice(object):

    """
    Numpy slice interface to Yt Fixed Resolution Buffers.

    YtSlice objects can be indexed using multidimensional slices::

        obj[1::3, 2::4]

    And the appropriate array will be extracted using Yt's Fixed
    Resolution Buffer.
    """

    def __init__(self, pf, field):
        """
        Parameters:
        -----------
        pf : Yt Data Object
           The Yt data to slice into
        field : str
           The name of the Yt field to extract
        """
        self.pf = pf
        self.field = field

    def _slice_args(self, view):
        index, coord = [(i, v) for i, v in enumerate(view)
                        if not isinstance(v, slice)][0]
        coord = 1. * coord / (self.pf.domain_dimensions[index] - 1)
        if coord == 1:
            coord = 1 - 1e-6
        return index, coord

    def _frb_args(self, view):
        sx = sy = None
        dim = self.pf.domain_dimensions
        for i, v in enumerate(view):
            if not isinstance(v, slice):
                continue

            start = 0 if v.start is None else v.start
            stop = dim[i] if v.stop is None else v.stop
            step = 1 if v.step is None else v.step
            s = slice(start, stop, step)
            if sx is None:
                sx = s
                nx = dim[i]
            else:
                sy = s
                ny = dim[i]

        l, r = sx.start, sx.stop
        b, t = sy.start, sy.stop
        w = int(np.ceil(1. * (r - l) / sx.step))
        h = int(np.ceil(1. * (t - b) / sy.step))
        #r = l + (step - 1) * w
        #t = b + (step - 1) * h
        bounds = (1. * l / nx, 1. * r / nx, 1. * b / ny, 1. * t / ny)
        return bounds, (h, w)

    def __getitem__(self, view):
        """
        Extract a slice, using numpy slice syntax

        Parameters
        ----------
        view : tuple of slices or integers

        Returns
        -------
        A numpy array
        """
        sl = self.pf.h.slice(*self._slice_args(view))
        frb = FixedResolutionBuffer(sl, *self._frb_args(view))
        return np.array(frb[self.field]).T


class TestSlice(object):

    def setup_method(self, method):
        x = np.arange(64).reshape((4, 4, 4))
        data = dict(data=x)
        y = load_uniform_grid(data, x.shape, 1)
        self.x = x
        self.y = y

    @pytest.mark.parametrize('slc', (np.s_[0, :, :],
                             np.s_[1, :, :],
                             np.s_[2, :, :],
                             np.s_[:, 2, :],
                             np.s_[0, :, 0:1],
                             np.s_[0:1, 0:1, 0],
                             np.s_[1, 0:2, 0:2]))
    def test_nostride(self, slc):
        w = YtSlice(self.y, 'data')
        np.testing.assert_array_almost_equal(w[slc], self.x[slc])

    def test_downsample_axis3(self):
        w = YtSlice(self.y, 'data')
        w = w[0, :, ::2]
        x = (self.x[0, :, ::2] + self.x[0, :, 1::2]) / 2.
        np.testing.assert_array_almost_equal(w, x)

    def test_downsample_axis2(self):
        w = YtSlice(self.y, 'data')
        w = w[0, ::2, :]
        x = (self.x[0, ::2, :] + self.x[0, 1::2, :]) / 2.
        np.testing.assert_array_almost_equal(w, x)

    def test_downsample_axis1(self):
        w = YtSlice(self.y, 'data')
        w = w[::2, :, 0]
        x = (self.x[::2, :, 0] + self.x[1::2, :, 0]) / 2.
        np.testing.assert_array_almost_equal(w, x)

    def test_uneven_downsample(self):
        w = YtSlice(self.y, 'data')
        w = w[0, :, ::3]
        x = (self.x[0, :, 1::2] + self.x[0, :, ::2]) / 2.
        np.testing.assert_array_almost_equal(w, x)

    def test_double_downsample(self):
        w = YtSlice(self.y, 'data')
        w = w[0, ::2, ::2]
        x = (self.x[0, ::2, ::2] + self.x[0, 1::2, 1::2] +
             self.x[0, ::2, 1::2] + self.x[0, 1::2, ::2]) / 4.
        np.testing.assert_array_almost_equal(w, x)
