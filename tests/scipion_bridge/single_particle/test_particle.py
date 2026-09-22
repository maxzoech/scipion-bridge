"""Tests for single_particle struct definitions: CTF, Coordinate, Particle, FlexParticle, Class2D."""

import pytest
import numpy as np

import scipion_bridge as B
from scipion_bridge.single_particle import (
    CTF,
    Coordinate,
    Particle,
    FlexParticle,
    Class2D,
)
from scipion_bridge.core.struct.storage import UninitializedFieldError


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


def test_ctf_schema():
    schema = CTF.schema()
    expected_fields = {"defocus_u", "defocus_v", "defocus_angle", "phase_shift", "resolution", "fit_quality"}
    expected_fields = {
        "defocus_u",
        "defocus_v",
        "defocus_angle",
        "phase_shift",
        "resolution",
        "fit_quality",
    }
    assert set(schema.fields.keys()) == expected_fields


def test_coordinate_schema():
    schema = Coordinate.schema()
    assert set(schema.fields.keys()) == {"x", "y"}


def test_particle_schema():
    schema = Particle.schema()
    assert set(schema.fields.keys()) == {"pixels", "ctf", "coordinate", "sampling_rate"}


def test_flex_particle_inherits_particle_fields():
    schema = FlexParticle.schema()
    assert "pixels" in schema.fields
    assert "ctf" in schema.fields
    assert "coordinate" in schema.fields
    assert "sampling_rate" in schema.fields
    assert "embeddings" in schema.fields


def test_class2d_schema():
    schema = Class2D.schema()
    assert set(schema.fields.keys()) == {"particles", "class_id", "representative"}


def test_class2d_has_no_label_field():
    schema = Class2D.schema()
    assert "label" not in schema.fields


# ---------------------------------------------------------------------------
# item_type tests
# ---------------------------------------------------------------------------


def test_set_particle_item_type():
    assert B.Set[Particle].item_type() is Particle


def test_set_class2d_item_type():
    assert B.Set[Class2D].item_type() is Class2D


def test_set_flex_particle_item_type():
    assert B.Set[FlexParticle].item_type() is FlexParticle


# ---------------------------------------------------------------------------
# Set[Particle] storage tests
# ---------------------------------------------------------------------------


def test_set_particle_pixels_write_read():
    pixels = np.random.rand(4, 64, 64).astype(np.float32)
    s = B.Set[Particle](capacity=4)
    s["pixels"] = pixels
    assert np.allclose(s["pixels"], pixels)


def test_set_particle_nested_ctf_write_read():
    n = 3
    s = B.Set[Particle](capacity=n)
    defocus_u = np.array([1.0, 2.0, 3.0]).reshape(n, 1)
    s["ctf"]["defocus_u"] = defocus_u
    assert np.allclose(s["ctf"]["defocus_u"], defocus_u)


def test_set_particle_nested_coordinate_write_read():
    n = 5
    s = B.Set[Particle](capacity=n)
    xs = np.arange(n, dtype=float).reshape(n, 1)
    ys = np.arange(n, 2 * n, dtype=float).reshape(n, 1)
    s["coordinate"]["x"] = xs
    s["coordinate"]["y"] = ys

    assert np.allclose(s["coordinate"]["x"], xs)
    assert np.allclose(s["coordinate"]["y"], ys)


def test_set_particle_indexed_access():
    n = 4
    s = B.Set[Particle](capacity=n)
    pixels = np.random.rand(n, 16, 16).astype(np.float32)
    s["pixels"] = pixels

    sample = s[2]
    assert np.allclose(sample.pixels, pixels[2])


def test_set_particle_ctf_indexed_access():
    n = 4
    s = B.Set[Particle](capacity=n)
    defocus_u = np.array([10.0, 20.0, 30.0, 40.0]).reshape(n, 1)
    s["ctf"]["defocus_u"] = defocus_u

    sample = s[1]
    assert sample.ctf.defocus_u == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# UninitializedFieldError tests
# ---------------------------------------------------------------------------


def test_uninitialized_ctf_raises():
    s = B.Set[Particle](capacity=2)
    pixels = np.zeros((2, 8, 8), dtype=np.float32)
    s["pixels"] = pixels

    with pytest.raises(UninitializedFieldError):
        _ = s[0].ctf.defocus_u


def test_uninitialized_coordinate_raises():
    s = B.Set[Particle](capacity=2)
    pixels = np.zeros((2, 8, 8), dtype=np.float32)
    s["pixels"] = pixels

    with pytest.raises(UninitializedFieldError):
        _ = s[0].coordinate.x


def test_uninitialized_sampling_rate_raises():
    s = B.Set[Particle](capacity=2)
    pixels = np.zeros((2, 8, 8), dtype=np.float32)
    s["pixels"] = pixels

    with pytest.raises(UninitializedFieldError):
        _ = s[0].sampling_rate


# ---------------------------------------------------------------------------
# Class2D struct tests
# ---------------------------------------------------------------------------


def test_class2d_construction():
    n_particles = 3
    particles_set = B.Set[Particle](capacity=n_particles)
    pixels = np.zeros((n_particles, 16, 16), dtype=np.float32)
    particles_set["pixels"] = pixels

    cls = Class2D(particles=particles_set, class_id=7)
    assert cls.class_id == 7


# ---------------------------------------------------------------------------
# Set[Class2D] ragged construction tests
# ---------------------------------------------------------------------------


def _make_particles(n: int, size: int = 16) -> B.Set[Particle]:
    """Helper: return a Set[Particle] of capacity *n* with random pixels."""
    s = B.Set[Particle](capacity=n)
    s["pixels"] = np.random.rand(n, size, size).astype(np.float32)
    return s


def test_set_class2d_sequence_construction():
    cls1 = Class2D(particles=_make_particles(5), class_id=1)
    cls2 = Class2D(particles=_make_particles(3), class_id=2)

    classes = B.Set[Class2D]([cls1, cls2])
    assert len(classes) == 2


def test_set_class2d_item_type_after_construction():
    cls1 = Class2D(particles=_make_particles(2), class_id=1)
    classes = B.Set[Class2D]([cls1])
    assert len(classes) == 1
    assert B.Set[Class2D].item_type() is Class2D


def test_set_class2d_class_id_column():
    cls1 = Class2D(particles=_make_particles(4), class_id=10)
    cls2 = Class2D(particles=_make_particles(6), class_id=20)

    classes = B.Set[Class2D]([cls1, cls2])
    ids = np.array(classes["class_id"]).squeeze()
    assert ids[0] == 10
    assert ids[1] == 20

