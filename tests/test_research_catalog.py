"""The research navigator must remain read-only and independent of ML imports."""

import copy
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/research_catalog.py"
SPEC = importlib.util.spec_from_file_location("research_catalog", SCRIPT)
catalog_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(catalog_tool)


@pytest.fixture
def fixture_catalog(tmp_path):
    (tmp_path / "doc.md").write_text("# Fixture\n")
    (tmp_path / "array.npz").write_bytes(b"not a real array; never parse")
    return {
        "schema_version": 1,
        "routes": {key: {"doc": "doc.md"} for key in ("A", "B", "C")},
        "core_docs": ["doc.md"],
        "forbidden_input_paths": [str(tmp_path / "unsafe")],
        "datasets": [
            {
                "id": "D1",
                "name": "fixture",
                "status": "historical_data",
                "routes": ["A"],
                "refs": ["doc.md"],
                "local_paths": ["array.npz"],
                "shape": "metadata only",
                "split": "DEV",
                "limits": ["No array reads"],
                "fit": {"A": "adapt", "B": "unsupported", "C": "needs_generation"},
            }
        ],
        "components": [],
        "evidence": [],
    }


def test_repository_catalog():
    catalog = catalog_tool.read_catalog()
    assert catalog_tool.validate_catalog(catalog) == []
    assert catalog_tool.markdown_link_errors(catalog["core_docs"]) == []


def test_valid_catalog_and_stat_only(fixture_catalog, tmp_path, monkeypatch):
    def forbidden_read(*args, **kwargs):
        raise AssertionError("validator must not read payload files")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    monkeypatch.setattr(Path, "read_text", forbidden_read)
    assert catalog_tool.validate_catalog(fixture_catalog, tmp_path, local_data=True) == []


def test_missing_local_data_is_optional(fixture_catalog, tmp_path):
    fixture_catalog["datasets"][0]["local_paths"] = ["missing.npz"]
    assert catalog_tool.validate_catalog(fixture_catalog, tmp_path) == []
    assert (
        "missing local data"
        in catalog_tool.validate_catalog(fixture_catalog, tmp_path, local_data=True)[0]
    )


@pytest.mark.parametrize("field", ["split", "shape", "limits", "local_paths"])
def test_requires_dataset_caveats(fixture_catalog, tmp_path, field):
    del fixture_catalog["datasets"][0][field]
    assert any(
        f"missing {field}" in e for e in catalog_tool.validate_catalog(fixture_catalog, tmp_path)
    )


def test_duplicate_ids(fixture_catalog, tmp_path):
    fixture_catalog["datasets"].append(copy.deepcopy(fixture_catalog["datasets"][0]))
    assert any(
        "duplicate id" in e for e in catalog_tool.validate_catalog(fixture_catalog, tmp_path)
    )


def test_unknown_route(fixture_catalog, tmp_path):
    fixture_catalog["datasets"][0]["routes"] = ["D"]
    assert any(
        "invalid routes" in e for e in catalog_tool.validate_catalog(fixture_catalog, tmp_path)
    )


def test_forbidden_path_even_without_loading(fixture_catalog, tmp_path):
    fixture_catalog["datasets"][0]["local_paths"] = [str(tmp_path / "unsafe" / "truth.npz")]
    assert any(
        "forbidden input path" in e
        for e in catalog_tool.validate_catalog(fixture_catalog, tmp_path)
    )


def test_ref_cannot_escape_repo(fixture_catalog, tmp_path):
    fixture_catalog["datasets"][0]["refs"] = ["../outside.md"]
    assert any(
        "escapes root" in e for e in catalog_tool.validate_catalog(fixture_catalog, tmp_path)
    )


def test_local_link_validation(tmp_path):
    (tmp_path / "ok.md").write_text("# Good\n")
    fence = chr(96) * 3
    (tmp_path / "a.md").write_text(
        "[ok](ok.md) [anchor](#hello) [external](https://example.com/a)\n"
        f"[bad](missing.md)\n{fence}\n[example](ignored.md)\n{fence}\n"
    )
    assert catalog_tool.markdown_link_errors(["a.md"], tmp_path) == [
        "a.md: missing link target missing.md"
    ]


def test_route_filter(fixture_catalog):
    assert len(catalog_tool.selected(fixture_catalog, "datasets", "A")) == 1
    assert catalog_tool.selected(fixture_catalog, "datasets", "B") == []
