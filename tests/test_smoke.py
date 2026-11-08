"""
Smoke test: imports every module. Fails if any module has a syntax or import error.
Runs before every commit via `pytest tests/`.
"""

def test_import_src_package():
    import src  # noqa: F401


def test_import_torque_profile():
    from src import torque_profile  # noqa: F401
    assert len(torque_profile.PARAM_NAMES) == 4
    assert torque_profile.PARAM_BOUNDS.shape == (4, 2)
