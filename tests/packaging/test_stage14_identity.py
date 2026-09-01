"""Regression guards for the public StructField research identity."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RESEARCH_FILENAMES = (
    "research_question.md",
    "method.md",
    "structural_field.md",
    "uncertainty.md",
    "reliability_gating.md",
    "equivariance.md",
    "benchmark.md",
    "experimental_protocol.md",
    "related_work.md",
)
FLOW = (
    "Image",
    "Probabilistic Structural Field",
    "Structural Uncertainty",
    "Reliability Gate",
    "Task Backbone",
    "Prediction",
)
LEGACY_TASK_TERMS = ("Q1", "Q2", "Q3", "数学建模", "比赛")
UNSUPPORTED_CLAIM = re.compile(
    r"\b(?:first-ever|world\s+first|SOTA|novel\s+proven)\b",
    flags=re.IGNORECASE,
)


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _assert_legacy_terms_are_historical(text: str) -> None:
    in_historical_section = False
    for line in text.splitlines():
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            in_historical_section = "Historical Motivation" in heading or "历史动机" in heading
        for term in LEGACY_TASK_TERMS:
            if term in line:
                assert in_historical_section, f"legacy task term {term!r} appears outside Historical Motivation"


def test_required_research_documents_exist() -> None:
    research_dir = ROOT / "docs" / "research"
    assert research_dir.is_dir()
    assert tuple(sorted(path.name for path in research_dir.glob("*.md"))) == tuple(sorted(RESEARCH_FILENAMES))


@pytest.mark.parametrize("readme", ("README.md", "README_CN.md"))
def test_readme_uses_structfield_research_identity(readme: str) -> None:
    text = _read(readme)
    assert "StructField-Vision" in text
    assert "1.0.0" in text
    _assert_legacy_terms_are_historical(text)


def test_research_docs_have_no_legacy_task_organization() -> None:
    for filename in RESEARCH_FILENAMES:
        text = _read(f"docs/research/{filename}")
        for term in LEGACY_TASK_TERMS:
            assert term not in text, f"{filename} contains legacy task term {term!r}"


def test_public_research_docs_avoid_unsupported_claims() -> None:
    paths = [ROOT / "README.md", ROOT / "README_CN.md"]
    paths.extend(ROOT / "docs" / "research" / name for name in RESEARCH_FILENAMES)
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert UNSUPPORTED_CLAIM.search(text) is None, path
        assert (
            re.search(
                r"^#{1,6}\s+V0\.2 performance\s*$",
                text,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            is None
        ), path


def test_research_question_and_hypothesis_labels_are_explicit() -> None:
    question = _read("docs/research/research_question.md")
    assert "When should neural networks trust explicit structural priors?" in question
    combined = "\n".join(_read(f"docs/research/{filename}") for filename in RESEARCH_FILENAMES)
    assert "Research Hypothesis" in combined
    assert "contains_results: false" in combined
    assert re.search(r"formal long(?:-| )(?:duration )?training", combined.lower())


def test_citation_identity_matches_distribution() -> None:
    citation = _read("CITATION.cff")
    assert 'title: "StructField-Vision"' in citation
    assert "version: 1.0.0" in citation
    assert 'license: "Apache-2.0"' in citation
