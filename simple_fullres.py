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
        self._grids = {}

    @property
    def shape(self):
        shp = self.ds.refine_by**self.ds.index.max_level*self.ds.domain_dimensions
        return tuple(shp.astype("int"))

    def _frb_args(self, view, axis):
        ix = self.ds.coordinates.x_axis[axis]
        iy = self.ds.coordinates.y_axis[axis]
        sx = view[ix]
        sy = view[iy]
        l, r = sx.start, sx.stop
        b, t = sy.start, sy.stop
        w = _steps(sx)
        h = _steps(sy)
        bounds = (self._dds[ix] * l + self._left_edge[ix],
                  self._dds[ix] * r + self._left_edge[ix],
                  self._dds[iy] * b + self._left_edge[iy],
                  self._dds[iy] * t + self._left_edge[iy])
        return bounds, (h, w)

    def _slice_args(self, view):
        index, coord = [(i, v) for i, v in enumerate(view)
                        if not isinstance(v, slice)][0]
        coord = self.get_loc(coord)[index]
        return index, coord

    def get_loc(self, idx):
        return self._left_edge + idx*self._dds

    def get_view(self, start, stop, shape):
        key = (start, stop, shape)
        if key not in self._grids:
            stt = self.get_loc(start)
            stp = self.get_loc(stop)
            self._grids[key] = self.ds.r[stt[0]:stp[0]:shape[0]*1j,
                                         stt[1]:stp[1]:shape[1]*1j,
                                         stt[2]:stp[2]:shape[2]*1j]
        return self._grids[key]

    _last_view = None
    _last_result = None

    def __getitem__(self, view):
        if self._last_view == view:
            return self._last_result
        self._last_view = view
        nd = len([v for v in view if isinstance(v, slice)])
        if nd == 3:
            print(view)
            vstart = np.zeros(3, dtype="int")
            vstop = np.zeros(3, dtype="int")
            vstep = np.zeros(3, dtype="int")
            for i, (v, shp) in enumerate(zip(view, self.shape)):
                ii = v.indices(shp)
                vstart[i] = ii[0]
                vstop[i] = ii[1]
                vstep[i] = ii[2]
            gdims = tuple(np.ceil((vstop-vstart)/vstep).astype("int"))
            vstart = tuple(vstart)
            vstop = tuple(vstop)
            if np.prod(gdims) == 1:
                loc = self.get_loc(np.array(vstart)+0.5)
                fdv = self.ds.find_field_values_at_point(self.field, loc)[:1]
                self._last_result = fdv.reshape(1,1,1)
            else:
                self._last_result = self.get_view(vstart, vstop, gdims)[self.field].d
        elif nd == 2:
            axis, coord = self._slice_args(view)
            sl = self.ds.slice(axis, coord)
            frb = FixedResolutionBuffer(sl, *self._frb_args(view, axis))
            self._last_result = frb[self.field].d.T
        else:
            raise RuntimeError
        return self._last_result

