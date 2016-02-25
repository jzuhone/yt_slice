from glue.core import Component
from glue.core import Data, DataCollection
from glue.qt.glue_application import GlueApplication
import numpy as np
import yt
from simple_fullres import ytSliceComponent
ds = yt.load("GasSloshing/sloshing_nomag2_hdf5_plt_cnt_0150")
d = Data(label=str(ds))
for field in ("density", "temperature", "velocity_x",
              "velocity_y", "velocity_z"):
    d.add_component(ytSliceComponent(ds, field), label=field)
dc = DataCollection(d)
ga = GlueApplication(dc)
ga.start()
