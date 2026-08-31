"""Run the hand-derived synthetic checks under pytest as well as via the CLI."""

from sqe.selftest import run_orientation, run_square, run_studio


def _report(checks):
    return "\n".join(f"  {c.name}\n      {c.detail}"
                     for c in checks if not c.passed)


def test_studio_frame_expectations():
    checks = run_studio(verbose=False)
    failed = [c for c in checks if not c.passed]
    assert not failed, "\n" + _report(checks)


def test_hostile_room_and_ambiguity():
    checks = run_square(verbose=False)
    failed = [c for c in checks if not c.passed]
    assert not failed, "\n" + _report(checks)


def test_front_estimation_against_ground_truth():
    checks = run_orientation(verbose=False)
    failed = [c for c in checks if not c.passed]
    assert not failed, "\n" + _report(checks)
