from pathlib import Path
import subprocess

import fishsuite
import pytest

from fishsuite.core import repro


@pytest.mark.parametrize("status, dirty", [("", "false"), ("?? new.py\n", "true")])
def test_versions_records_imported_source_identity(tmp_path, monkeypatch, status, dirty):
    package_dir = tmp_path / "imported" / "fishsuite"
    monkeypatch.setattr(fishsuite, "__file__", str(package_dir / "__init__.py"))
    calls = []

    def run(command, **kwargs):
        calls.append((list(command), kwargs))
        output = "a" * 40 + "\n" if "rev-parse" in command else status
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    assert repro.write_versions_txt(tmp_path, 0)
    lines = (tmp_path / "versions.txt").read_text(encoding="utf-8").splitlines()
    assert any(l.startswith("fishsuite_git_commit: " + "a" * 40) for l in lines)
    assert f"fishsuite_git_dirty: {dirty}" in lines
    assert f"fishsuite_source_path: {package_dir.resolve()}" in lines
    expected_prefix = ["git", "-C", str(package_dir.resolve())]
    assert expected_prefix + ["rev-parse", "HEAD"] in [call[0] for call in calls]
    assert expected_prefix + ["status", "--porcelain", "--untracked-files=no"] in [call[0] for call in calls]
    assert all(kwargs["timeout"] <= 10 for command, kwargs in calls if command[0] == "git")


@pytest.mark.parametrize("error", [
    FileNotFoundError("git is absent"),
    subprocess.TimeoutExpired("git", 5),
    subprocess.CalledProcessError(128, "git"),
])
def test_versions_survives_unavailable_git(tmp_path, monkeypatch, error):
    def run(*args, **kwargs):
        raise error

    monkeypatch.setattr(subprocess, "run", run)
    assert repro.write_versions_txt(tmp_path, 0)
    lines = (tmp_path / "versions.txt").read_text(encoding="utf-8").splitlines()
    assert "fishsuite_git_commit: unknown" in lines
    assert "fishsuite_git_dirty: UNKNOWN" in lines
    assert f"fishsuite_source_path: {Path(fishsuite.__file__).resolve().parent}" in lines
