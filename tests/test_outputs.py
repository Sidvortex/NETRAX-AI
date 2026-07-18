import json

from netrax.outputs.jsonl import JsonlWriter


def test_jsonl_appends_and_respects_toggle(settings):
    w = JsonlWriter(settings)
    w({"type": "gesture", "label": "peace"})
    w({"type": "object", "label": "cup"})
    lines = w.path.read_text().splitlines()
    assert [json.loads(x)["label"] for x in lines] == ["peace", "cup"]
    settings.update({"outputs": {"jsonl": {"enabled": False}}}, save=False)
    w({"type": "x", "label": "y"})
    assert len(w.path.read_text().splitlines()) == 2
