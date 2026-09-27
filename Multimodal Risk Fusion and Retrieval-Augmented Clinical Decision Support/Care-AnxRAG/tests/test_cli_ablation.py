from types import SimpleNamespace

from typer.testing import CliRunner

from care_anxrag import cli


runner = CliRunner()


def test_evaluate_ablation_cli_outputs_named_reports(monkeypatch, tmp_path) -> None:
    benchmark = tmp_path / "benchmark.jsonl"
    benchmark.write_text(
        '{"id":"q1","question":"test question"}\n',
        encoding="utf-8",
    )

    runtime = SimpleNamespace(
        retriever=object(),
        rag=object(),
    )
    monkeypatch.setattr(cli, "_runtime", lambda project_root=None: runtime)

    class Report:
        def as_dict(self):
            return {"count": 1, "mrr": 0.5}

    monkeypatch.setattr(
        cli,
        "run_ablation",
        lambda retriever, rag, items: {
            "B0_dense_only": Report(),
            "CARE_full": Report(),
        },
    )

    result = runner.invoke(
        cli.app,
        ["evaluate-ablation", str(benchmark)],
    )

    assert result.exit_code == 0
    assert '"B0_dense_only"' in result.stdout
    assert '"CARE_full"' in result.stdout
