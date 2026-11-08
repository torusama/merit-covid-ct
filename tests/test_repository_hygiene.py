"""Maintain documentation language, local references and repository entry points."""

import csv
import importlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import pytest


ROOT = Path(__file__).resolve().parents[1]
VIETNAMESE = re.compile(
    r"[\u00c0\u00c1\u00c2\u00c3\u00c8\u00c9\u00ca\u00cc\u00cd"
    r"\u00d2\u00d3\u00d4\u00d5\u00d9\u00da\u00dd"
    r"\u00e0\u00e1\u00e2\u00e3\u00e8\u00e9\u00ea\u00ec\u00ed"
    r"\u00f2\u00f3\u00f4\u00f5\u00f9\u00fa\u00fd"
    r"\u0102\u0103\u0110\u0111\u0128\u0129\u0168\u0169"
    r"\u01a0\u01a1\u01af\u01b0\u1ea0-\u1ef9]"
)


def maintained_markdown():
    paths = [ROOT / "README.md", ROOT / "CHANGELOG.md"]
    for directory in ("docs", "src", "ct_pipeline", "analysis"):
        paths.extend((ROOT / directory).rglob("*.md"))
    return sorted(paths)


def markdown_anchors(text):
    anchors, seen = set(), {}
    for line in text.splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match:
            title = match.group(1).lower().replace(" ", "-")
            slug = re.sub(r"[^\w-]", "", title)
            number = seen.get(slug, 0)
            anchors.add(f"{slug}-{number}" if number else slug)
            seen[slug] = number + 1
    return anchors


def test_maintained_prose_and_project_python_are_english():
    paths = maintained_markdown()
    for directory in ("src", "scripts", "tools", "examples", "tests", "ct_pipeline"):
        paths.extend((ROOT / directory).rglob("*.py"))
    paths.extend((ROOT / "configs").glob("*.yaml"))
    unexpected = [
        str(path.relative_to(ROOT))
        for path in paths
        if VIETNAMESE.search(path.read_text(encoding="utf-8-sig"))
    ]
    assert not unexpected, f"Non-English project prose: {unexpected}"


def test_local_markdown_targets_and_anchors_exist():
    failures = []
    for path in maintained_markdown():
        text = path.read_text(encoding="utf-8-sig")
        for link in re.findall(r"!?\[[^\]\n]*\]\(([^)\n]+)\)", text):
            url = urlsplit(link.strip("<>"))
            if url.scheme or url.netloc:
                continue
            target = (path.parent / unquote(url.path)).resolve() if url.path else path
            if not target.exists():
                failures.append(f"{path.relative_to(ROOT)} -> {link}")
            elif url.fragment and target.suffix.lower() == ".md":
                if unquote(url.fragment) not in markdown_anchors(target.read_text(encoding="utf-8-sig")):
                    failures.append(f"{path.relative_to(ROOT)} -> missing anchor {link}")
    assert not failures, "\n".join(failures)


def test_one_project_readme_and_all_topic_guides_are_linked():
    paths = maintained_markdown()
    assert [p.relative_to(ROOT).as_posix() for p in paths if p.name.lower() == "readme.md"] == ["README.md"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for guide in (ROOT / "docs").glob("*.md"):
        assert f"docs/{guide.name}" in readme, guide.name


def test_saved_primary_csv_schema_is_fully_documented():
    with (ROOT / "analysis/covid19_20/case_statistics.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        fields = csv.DictReader(handle).fieldnames
    assert len(fields) == 71
    guide = (ROOT / "docs/dataset_reports.md").read_text(encoding="utf-8")
    assert all(f"`{field}`" in guide for field in fields)


def test_pytest_has_one_configuration_source(request):
    assert not (ROOT / "pytest.ini").exists()
    assert request.config.inipath == ROOT / "pyproject.toml"
    assert request.config.getini("testpaths") == ["tests"]
    assert request.config.getini("pythonpath") == [ROOT, ROOT / "src"]
    assert request.config.getoption("importmode") == "importlib"


def report_writer():
    pytest.importorskip("torch")
    pytest.importorskip("matplotlib")
    return importlib.import_module("tools.summarize_covid19_20").write_report


def test_report_generator_uses_english_and_supplied_statistics(tmp_path):
    summary = {
        "cases_using_assumed_mm": [],
        "spacing_z_mm_counts": {"3": 2},
        "generated_at_bangkok": "synthetic-fixture",
        "source_audit_config": {"intensity_encoding": "native", "hu_window": None},
        "cases": 2, "slices": 8, "positive_slices": 3, "empty_slices": 5,
        "slice_count_min": 4, "slice_count_max": 4, "slice_count_median": 4,
        "lesion_voxels": 12, "total_voxels": 128,
        "lesion_fraction_volume_pooled": 12 / 128,
        "lesion_fraction_volume_macro": 12 / 128,
        "ct_scaled_intensity_range": [-1000, 400],
    }
    report_writer()(summary, tmp_path)
    text = (tmp_path / "REPORT.md").read_text(encoding="utf-8")
    assert not VIETNAMESE.search(text)
    assert "2 CT volumes" in text and "12 lesion voxels / 128 total CT voxels" in text
    assert "| 3 | 2 |" in text
    assert "Unknown-unit cases explicitly assumed to be mm: none" in text
    assert "not model predictions" in text


def test_stored_report_matches_english_generator_without_reauditing_data(tmp_path):
    summary = json.loads(
        (ROOT / "analysis/covid19_20/summary.json").read_text(encoding="utf-8")
    )
    report_writer()(summary, tmp_path)
    assert (tmp_path / "REPORT.md").read_text(encoding="utf-8") == (
        ROOT / "analysis/covid19_20/REPORT.md"
    ).read_text(encoding="utf-8")
