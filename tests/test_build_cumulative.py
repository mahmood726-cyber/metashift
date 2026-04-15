import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_cumulative import main, resolve_paths


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_main_uses_repo_relative_sibling_projects(tmp_path):
    projects_root = tmp_path / 'projects'
    project_root = projects_root / 'MetaShift'
    project_root.mkdir(parents=True)

    paths = resolve_paths(project_root=project_root, projects_root=projects_root)

    paths['validation_inputs'].parent.mkdir(parents=True, exist_ok=True)
    paths['validation_inputs'].write_text(
        json.dumps([{
            'review_id': 'CD000001',
            'analysis_name': 'Synthetic trajectory',
            'yi': [0.1, 0.15, 0.2],
            'sei': [0.1, 0.1, 0.1],
        }]),
        encoding='utf-8',
    )
    write_csv(
        paths['fragility_results'],
        [
            {'review_id': 'CD000001', 'analysis_name': 'Synthetic trajectory', 'k': '3'},
            {'review_id': 'CD000002', 'analysis_name': 'Aggregate only', 'k': '5'},
        ],
        ['review_id', 'analysis_name', 'k'],
    )
    write_csv(
        paths['prediction_results'],
        [
            {'review_id': 'CD000001', 'theta': '0.2', 'p_value': '0.01', 'I2': '10', 'tau2': '0.02'},
            {'review_id': 'CD000002', 'theta': '0.3', 'p_value': '0.04', 'I2': '20', 'tau2': '0.03'},
        ],
        ['review_id', 'theta', 'p_value', 'I2', 'tau2'],
    )

    out_path = main(project_root=project_root, projects_root=projects_root)

    payload = json.loads(out_path.read_text(encoding='utf-8'))
    assert out_path == project_root / 'data' / 'cumulative.json'
    assert payload['corpus_summary'] == {
        'n_with_trajectory': 1,
        'n_aggregate_only': 1,
        'n_total': 2,
    }
    assert payload['reviews_with_trajectory'][0]['review_id'] == 'CD000001'
    assert payload['reviews_aggregate_only'][0]['review_id'] == 'CD000002'
