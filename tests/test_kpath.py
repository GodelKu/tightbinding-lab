import math

import numpy as np
import pytest

from tblab.kpath import (
    brillouin_zone_polygon,
    build_kpath,
    normalise_label,
    parse_path,
    polygon_area,
)
from tblab.models import Graphene, Square


def test_parse_path_aliases():
    assert parse_path("G-K-M-G") == ["G", "K", "M", "G"]
    assert parse_path("Γ - K' - M") == ["G", "Kp", "M"]
    assert parse_path("gamma,x,m") == ["G", "X", "M"]
    assert normalise_label("kp") == "Kp"


def test_parse_path_needs_two_points():
    with pytest.raises(ValueError):
        parse_path("G")


def test_build_kpath_hits_all_vertices_and_is_continuous():
    g = Graphene()
    kp = build_kpath(g, parse_path("G-K-M-G"), nk=240)
    assert kp.tick_labels == ["Γ", "K", "M", "Γ"]
    cart = g.cart(kp.kfrac)
    # cumulative distance equals the arc length of the Cartesian polyline
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(cart, axis=0), axis=1))])
    assert np.allclose(arc, kp.x, atol=1e-12)
    # the vertices are present in the sampled path
    for label, tick in zip(kp.labels, kp.tick_x, strict=True):
        i = int(np.argmin(np.abs(kp.x - tick)))
        assert np.allclose(kp.kfrac[i], g.special_points[label])
    assert 200 <= len(kp.x) <= 250


def test_graphene_path_segment_lengths_match_geometry():
    """Segment lengths from hand-derived geometry (bond length a, lattice constant L = sqrt(3) a):
    |ΓK| = 4π/(3L), |KM| = 2π/(3L), |ΓM| = 2π/(√3 L)."""
    a = 1.0
    lat = math.sqrt(3.0) * a
    g = Graphene(a=a)
    kp = build_kpath(g, ["G", "K", "M", "G"], nk=600)
    assert kp.tick_x[1] == pytest.approx(4.0 * math.pi / (3.0 * lat), rel=1e-12)
    assert kp.tick_x[2] - kp.tick_x[1] == pytest.approx(2.0 * math.pi / (3.0 * lat), rel=1e-12)
    assert kp.tick_x[3] - kp.tick_x[2] == pytest.approx(
        2.0 * math.pi / (math.sqrt(3.0) * lat), rel=1e-12
    )


def test_unknown_label_and_zero_segment():
    with pytest.raises(ValueError, match="unknown point"):
        build_kpath(Square(), ["G", "K"])
    with pytest.raises(ValueError, match="zero-length"):
        build_kpath(Square(), ["G", "G"])


@pytest.mark.parametrize("a", [1.0, 1.42, 2.5])
def test_bz_area_and_corners_graphene(a):
    g = Graphene(a=a)
    poly = brillouin_zone_polygon(g.reciprocal)
    assert len(poly) == 6
    area_cell = 3.0 * math.sqrt(3.0) * a * a / 2.0
    assert polygon_area(poly) == pytest.approx((2 * math.pi) ** 2 / area_cell, rel=1e-12)
    # K and K' are BZ corners: their length equals the hexagon circumradius 4π/(3√3 a)
    radius = 4.0 * math.pi / (3.0 * math.sqrt(3.0) * a)
    assert np.linalg.norm(poly, axis=1) == pytest.approx(radius, rel=1e-12)
    for name in ("K", "Kp"):
        assert np.linalg.norm(g.cart(np.array(g.special_points[name]))) == pytest.approx(
            radius, rel=1e-12
        )
    # M is the midpoint of an edge: its length is the inradius (√3/2) R
    m = np.linalg.norm(g.cart(np.array(g.special_points["M"])))
    assert m == pytest.approx(radius * math.sqrt(3.0) / 2.0, rel=1e-12)


def test_bz_area_square():
    sq = Square(a=2.0)
    poly = brillouin_zone_polygon(sq.reciprocal)
    assert polygon_area(poly) == pytest.approx((2 * math.pi / 2.0) ** 2)
