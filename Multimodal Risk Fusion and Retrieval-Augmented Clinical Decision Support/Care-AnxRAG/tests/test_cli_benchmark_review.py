from typer.testing import CliRunner

from care_anxrag.cli import app


def test_review_package_cli_needs_no_runtime(tmp_path):
    result = CliRunner().invoke(app, ["benchmark-review-package", str(tmp_path / "review")])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "review" / "test.csv").exists()


def test_split_validation_cli_rejects_empty_benchmarks(tmp_path):
    development = tmp_path / "dev.jsonl"
    test = tmp_path / "test.jsonl"
    development.write_text("")
    test.write_text("")
    result = CliRunner().invoke(app, ["benchmark-validate-splits", str(development), str(test)])
    assert result.exit_code == 1
    assert "no items" in result.output
