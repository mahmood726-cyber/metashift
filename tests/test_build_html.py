import json
from pathlib import Path

from build_html import main, render_html


TEMPLATE = """<!DOCTYPE html>
<html>
<body>
<div id="app"></div>
<script>
function init(){ return true; }
</script>
<script id="dataScript">
DATA = {"old": true};
</script>
</body>
</html>
"""


def test_main_rewrites_embedded_data_into_repo_relative_outputs(tmp_path):
    project_root = tmp_path / "MetaShift"
    (project_root / "data").mkdir(parents=True)
    (project_root / "e156-submission" / "assets").mkdir(parents=True)

    payload = {"corpus_summary": {"n_total": 2}, "reviews_with_trajectory": []}
    (project_root / "data" / "cumulative.json").write_text(json.dumps(payload), encoding="utf-8")
    (project_root / "metashift.html").write_text(TEMPLATE, encoding="utf-8")

    written = main(project_root=project_root)

    assert written == [
        project_root / "metashift.html",
        project_root / "e156-submission" / "assets" / "metashift.html",
    ]

    root_html = (project_root / "metashift.html").read_text(encoding="utf-8")
    asset_html = (project_root / "e156-submission" / "assets" / "metashift.html").read_text(encoding="utf-8")
    assert 'DATA = {"corpus_summary": {"n_total": 2}, "reviews_with_trajectory": []};' in root_html
    assert asset_html == root_html


def test_render_html_fails_closed_when_data_script_block_missing():
    try:
        render_html("<html><body><script>const x = 1;</script></body></html>", {"ok": True})
    except ValueError as exc:
        assert "dataScript" in str(exc)
    else:
        raise AssertionError("render_html should fail closed when the dataScript block is missing")
