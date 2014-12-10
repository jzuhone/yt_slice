from glue.core import Component
import numpy as np
from yt.visualization.fixed_resolution import FixedResolutionBuffer

def _steps(slice):
    return int(np.ceil(1. * (slice.stop - slice.start) / slice.step))

class ytSliceComponent(Component):
    ndim = 3
    def __init__(self, ds, field):
        self.ds = ds
        self.field = field
        self._dds = (ds.domain_width / self.shape).in_units("code_length").d
        self._left_edge = self.ds.domain_left_edge.in_units("code_length").d
        self._right_edge = self.ds.domain_right_edge.in_units("code_length").d

    @property
    def shape(self):
        return tuple(self.ds.refine_by**min(self.ds.max_level, 5)
                   * self.ds.domain_dimensions)

    def _frb_args(self, view, axis):
        dim = self.shape

        ix = self.ds.coordinates.x_axis[axis]
        iy = self.ds.coordinates.y_axis[axis]

        sx = view[ix]
        nx = dim[ix]
        sy = view[iy]
        ny = dim[iy]

        l, r = sx.start, sx.stop
        b, t = sy.start, sy.stop
        w = _steps(sx)
        h = _steps(sy)
        bounds = (1. * l / nx + self._left_edge[ix],
                  1. * r / nx + self._left_edge[ix],
                  1. * b / ny + self._left_edge[iy],
                  1. * t / ny + self._left_edge[iy])
        return bounds, (h, w)

    _last_view = None
    _last_result = None
    def __getitem__(self, view):
        if self._last_view == view:
            return self._last_result
        self._last_view = view
        i = len([v for v in view if isinstance(v, slice)])
        if i == 3:
            LE, RE = [], []
            for i, v in enumerate(view):
                c = v.start
                if c is None: c = 0.0
                LE.append(self._left_edge[i] + c * self._dds[i])
                c = v.stop
                if c is None: c = self.shape[i] - 1
                RE.append(self._left_edge[i] + c * self._dds[i])
            LE = np.array(LE)
            RE = np.array(RE)
            obj = self.ds.region((LE + RE)/2.0, LE, RE)
            self._last_result = obj[self.field]
        elif i == 2:
            axis, coord = self._slice_args(view)
            bounds, (h, w) = self._frb_args(view, axis)
            sl = self.ds.slice(axis, coord)
            frb = FixedResolutionBuffer(sl, *self._frb_args(view, axis))
            self._last_result = frb[self.field].d.T
        else:
            raise RuntimeError
        return self._last_result

    def _slice_args(self, view):
        index, coord = [(i, v) for i, v in enumerate(view)
                        if not isinstance(v, slice)][0]
        coord = coord * self._dds[index]
        return index, coord

