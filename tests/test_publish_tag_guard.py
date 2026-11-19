from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def publisher():
    here = Path(__file__).resolve()
    script = next(
        (
            candidate
            for candidate in (parent / "scripts" / "publish_public.py" for parent in here.parents)
            if candidate.is_file()
        ),
        None,
    )
    if script is None:
        pytest.skip("publisher is maintained in the development repository")
    spec = importlib.util.spec_from_file_location("publish_public", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_tag_guard_rejects_reused_tags_and_accepts_revision(publisher, tmp_path) -> None:
    remote = tmp_path / "remote.git"
    source = tmp_path / "source"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    subprocess.run(["git", "init", str(source)], check=True, capture_output=True)

    def git(*args):
        return subprocess.run(["git", "-C", str(source), *args], check=True, capture_output=True)

    git("-c", "user.name=Review Test", "-c", "user.email=review@example.invalid", "commit", "--allow-empty", "-m", "fixture")
    git("remote", "add", "origin", str(remote))
    git("tag", "v1.7.3")
    git("push", "origin", "refs/tags/v1.7.3")
    with pytest.raises(SystemExit, match="already published"):
        publisher._release_preflight(source, "1.7.3", None)
    with pytest.raises(SystemExit, match="must match"):
        publisher._release_preflight(source, "1.7.3", "v1.7.4")
    assert publisher._release_preflight(source, "1.7.3", "v1.7.3-r2") == "v1.7.3-r2"
    git("tag", "v1.7.3-r2")
    with pytest.raises(SystemExit, match="already exists locally"):
        publisher._release_preflight(source, "1.7.3", "v1.7.3-r2")
