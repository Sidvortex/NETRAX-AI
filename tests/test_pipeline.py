from netrax.core.pipeline import Pipeline
from netrax.core.scheduler import Scheduler

from .conftest import stub_factory

ORDER = ["face_detect", "hands", "pose", "objects"]


def make(settings, **mods):
    patch = {"modules": {m: {"enabled": False} for m in ("face_recog", "head_pose")}}
    for m, v in mods.items():
        patch["modules"][m] = v
    settings.update(patch, save=False)
    p = Pipeline(settings, factory=stub_factory, order=ORDER)
    p.sync(wait=True)
    return p


def test_scheduler_phases_spread_load():
    s = Scheduler(["a", "b"])
    a = [s.should_run("a", 2, i) for i in range(6)]
    b = [s.should_run("b", 2, i) for i in range(6)]
    assert a == [True, False] * 3
    assert b == [False, True] * 3
    assert all(s.should_run("a", 1, i) for i in range(5))


def test_every_n_reuses_last_results(settings, frame):
    p = make(settings, objects={"enabled": True, "every_n": 5})
    states = [p.process(frame) for _ in range(10)]
    det = p.detector("objects")
    assert det.calls == 3   # first frame (nothing cached yet) + 2 scheduled runs
    assert all(len(s.get("objects")) == 1 for s in states)   # cached between runs
    assert sum("objects" in s.ran for s in states) == 3
    assert p.detector("face_detect").calls == 10


def test_disabled_module_is_unloaded(settings, frame):
    p = make(settings, objects={"enabled": True})
    assert p.slots["objects"].state == "ready"
    settings.update({"modules": {"objects": {"enabled": False}}}, save=False)
    p.sync(wait=True)
    assert p.slots["objects"].state == "off"
    assert p.process(frame).get("objects") == []


def test_load_failure_is_reported_not_fatal(settings, frame):
    p = make(settings, hands={"enabled": True})
    assert p.slots["hands"].state == "error"
    assert "model missing" in p.slots["hands"].error
    assert len(p.process(frame).get("face_detect")) == 1


def test_crashing_module_does_not_stop_pipeline(settings, frame):
    p = make(settings, pose={"enabled": True, "every_n": 1})
    for _ in range(3):
        st = p.process(frame)
    assert st.get("pose") == []
    assert p.slots["pose"].stats.errors == 3
    assert "boom" in p.slots["pose"].error
    assert len(st.get("face_detect")) == 1


def test_stats_and_state_dict(settings, frame):
    p = make(settings)
    for _ in range(5):
        st = p.process(frame)
    s = p.status()["face_detect"]
    assert s["state"] == "ready" and s["runs"] == 5 and s["ms"] >= 0
    d = st.to_dict()
    assert d["faces"][0]["track_id"] == 1
    assert "raw" not in d["faces"][0]
    out = p.draw(frame.copy(), st)
    assert out.any()   # something was drawn
