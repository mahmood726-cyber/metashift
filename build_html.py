import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "cumulative.json"
APP_PATH = PROJECT_ROOT / "metashift.html"
ASSET_PATH = PROJECT_ROOT / "e156-submission" / "assets" / "metashift.html"
DATA_BLOCK = re.compile(r"(<script id=\"dataScript\">\s*)DATA = .*?;\s*(</script>)", re.DOTALL)


def resolve_paths(project_root=PROJECT_ROOT, data_path=None, template_path=None, output_paths=None):
    project_root = Path(project_root)
    data_path = Path(data_path) if data_path is not None else project_root / "data" / "cumulative.json"

    default_app = project_root / "metashift.html"
    default_asset = project_root / "e156-submission" / "assets" / "metashift.html"
    if template_path is None:
        if default_app.exists():
            template_path = default_app
        elif default_asset.exists():
            template_path = default_asset
        else:
            template_path = default_app
    else:
        template_path = Path(template_path)

    if output_paths is None:
        output_paths = [default_app, default_asset]
    else:
        output_paths = [Path(path) for path in output_paths]

    return {
        "project_root": project_root,
        "data_path": data_path,
        "template_path": template_path,
        "output_paths": output_paths,
    }


def render_html(template_text, payload):
    match = DATA_BLOCK.search(template_text)
    if match is None:
        raise ValueError("Template is missing the <script id=\"dataScript\"> DATA block.")

    embedded = "DATA = " + json.dumps(payload, ensure_ascii=False) + ";"
    return DATA_BLOCK.sub(r"\1" + embedded + r"\n\2", template_text, count=1)


def analyze_html(html):
    return {
        "div_opens": len(re.findall(r"<div[\s>]", html)),
        "div_closes": html.count("</div>"),
        "script_blocks": html.count("<script"),
        "data_block_present": bool(DATA_BLOCK.search(html)),
    }


def write_outputs(html, output_paths):
    written = []
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        written.append(path)
    return written


def main(project_root=PROJECT_ROOT, data_path=None, template_path=None, output_paths=None):
    paths = resolve_paths(
        project_root=project_root,
        data_path=data_path,
        template_path=template_path,
        output_paths=output_paths,
    )

    payload = json.loads(paths["data_path"].read_text(encoding="utf-8"))
    template_text = paths["template_path"].read_text(encoding="utf-8")
    html = render_html(template_text, payload)
    written = write_outputs(html, paths["output_paths"])
    diagnostics = analyze_html(html)

    print(f"Embedded {paths['data_path']} into {len(written)} HTML file(s).")
    for path in written:
        print(path)
    print(
        "Diagnostics: "
        f"divs open={diagnostics['div_opens']} close={diagnostics['div_closes']} "
        f"scripts={diagnostics['script_blocks']} data_block={diagnostics['data_block_present']}"
    )
    return written


if __name__ == "__main__":
    main()
