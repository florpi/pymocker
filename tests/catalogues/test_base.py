from pymocker.catalogues.halo import HaloCatalogue
import numpy as np
import pandas as pd


def test__from_frame():
    n = 10
    df = pd.DataFrame(
        {
            "x": np.random.random((n,)),
            "y": np.random.random((n,)),
            "z": np.random.random((n,)),
            "v_x": np.random.random((n,)),
            "v_y": np.random.random((n,)),
            "v_z": np.random.random((n,)),
            "mass": np.random.random((n,)),
        }
    )
    cat = HaloCatalogue.from_frame(df)
    np.testing.assert_equal(cat.pos[:, 0], df.x)
    np.testing.assert_equal(cat.pos[:, 1], df.y)
    np.testing.assert_equal(cat.pos[:, 2], df.z)

    np.testing.assert_equal(cat.vel[:, 0], df.v_x)
    np.testing.assert_equal(cat.vel[:, 1], df.v_y)
    np.testing.assert_equal(cat.vel[:, 2], df.v_z)

    np.testing.assert_equal(cat.mass, df.mass)
