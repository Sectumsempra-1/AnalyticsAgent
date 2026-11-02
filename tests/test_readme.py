"""Run the subgraph examples from README.md exactly as written, with a scripted model."""

import re
import shutil
from pathlib import Path

import pytest

import analytics_agent.graph as graph_module
from tests.conftest import SALES_AMOUNT_SUM
from tests.fake_model import ScriptedModel, tool_call

README = Path(__file__).parent.parent / "README.md"
SQL = "SELECT SUM(amount) AS total FROM sales"


def readme_examples() -> list[str]:
    text = README.read_text(encoding="utf-8")
    return re.findall(r"<!-- tested: tests/test_readme.py -->\s*```python\n(.*?)```", text, re.DOTALL)


def test_readme_has_two_tested_examples() -> None:
    assert len(readme_examples()) == 2


@pytest.mark.parametrize("index", [0, 1], ids=["call-from-node", "graph-as-node"])
def test_readme_subgraph_example_runs(
    index: int, tmp_path: Path, fixture_files: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # fmt: skip
    (tmp_path / "data").mkdir()
    shutil.copy(fixture_files["sales"], tmp_path / "data" / "sales.csv")
    monkeypatch.chdir(tmp_path)  # the examples use the relative path data/sales.csv
    for key in ("MODEL_PROVIDER", "MODEL_BASE_URL", "AGENT_OUTPUT_DIR"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("MODEL_NAME", "scripted")
    summary = f"Total sales are {SALES_AMOUNT_SUM}."
    model = ScriptedModel(
        turns=[tool_call("run_sql", query=SQL)],
        reports=[graph_module.AnalysisResult(summary=summary, confidence="high")],
    )
    monkeypatch.setattr(graph_module, "load_model", lambda settings: model)

    exec(compile(readme_examples()[index], f"README example {index + 1}", "exec"), {"__name__": "readme"})

    assert summary in capsys.readouterr().out
    assert model.bound_tools  # the agent really ran
