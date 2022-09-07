from operator import ge
from pymocker.catalogues.halo import HaloCatalogue
import numpy as np
import pandas as pd
import pytest


def generate_random_df(n):
    return pd.DataFrame(
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


@pytest.fixture(name="df", scope="module")
def create_df():
    n = 10
    return generate_random_df(n=n)


@pytest.fixture(name="cat", scope="module")
def create_cat(df):
    return HaloCatalogue.from_frame(df)


def test__from_frame(df, cat):
    np.testing.assert_equal(cat.pos[:, 0], df.x)
    np.testing.assert_equal(cat.pos[:, 1], df.y)
    np.testing.assert_equal(cat.pos[:, 2], df.z)

    np.testing.assert_equal(cat.vel[:, 0], df.v_x)
    np.testing.assert_equal(cat.vel[:, 1], df.v_y)
    np.testing.assert_equal(cat.vel[:, 2], df.v_z)

    np.testing.assert_equal(cat.mass, df.mass)


def test__to_frame(df, cat):
    df_from_cat = cat.to_frame()
    np.testing.assert_equal(df_from_cat.x.values, df.x.values)
    np.testing.assert_equal(df_from_cat.y.values, df.y.values)
    np.testing.assert_equal(df_from_cat.z.values, df.z.values)
    np.testing.assert_equal(df_from_cat.v_x.values, df.v_x.values)
    np.testing.assert_equal(df_from_cat.v_y.values, df.v_y.values)
    np.testing.assert_equal(df_from_cat.v_z.values, df.v_z.values)
    np.testing.assert_equal(df_from_cat.mass.values, df.mass.values)


def test__add_catalogues(
    cat,
):
    df = generate_random_df(n=20)
    second_catalogue = HaloCatalogue.from_frame(df)
    summed_catalogue = cat + second_catalogue
    assert len(summed_catalogue) == len(cat) + len(second_catalogue)
    np.testing.assert_equal(summed_catalogue.pos[: len(cat)], cat.pos)
    np.testing.assert_equal(summed_catalogue.pos[len(cat) :], second_catalogue.pos)

    np.testing.assert_equal(summed_catalogue.vel[: len(cat)], cat.vel)
    np.testing.assert_equal(summed_catalogue.vel[len(cat) :], second_catalogue.vel)

    np.testing.assert_equal(summed_catalogue.mass[: len(cat)], cat.mass)
    np.testing.assert_equal(summed_catalogue.mass[len(cat) :], second_catalogue.mass)
