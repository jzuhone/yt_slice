import dumb_glue
import yt
ds = yt.load("IsolatedGalaxy/galaxy0030/galaxy0030")
cg = ds.covering_grid(2, [0,0,0], [128, 128, 128])
data = cg["density"]
dumb_glue.export_glue(ds, data, "density")
