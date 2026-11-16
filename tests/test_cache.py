from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path

import pytest

import deeptutor.review.cache as cache_module
from deeptutor.review.cache import (
    DEFAULT_DIR_NAME,
    ENV_VAR,
    TEMP_FILE_MAX_AGE_SECONDS,
    DiskCache,
    cache_key,
)


def _tmp_suffix() -> str:
    """The suffix lives in a private module constant; read it, do not copy it."""
    return cache_module._TMP_SUFFIX

KEY = "0" * 64
OTHER_KEY = "1" * 64
SUFFIX = "svg"


def _entry(cache: DiskCache, key: str = KEY, suffix: str = SUFFIX) -> Path:
    return cache.root / f"{key}.{suffix}"


def _put_aged(
    cache: DiskCache, key: str, mtime_ns: int, *, data: bytes = b"payload"
) -> None:
    """Write ``key`` and pin its mtime so eviction order is deterministic."""
    cache.put(key, suffix=SUFFIX, data=data)
    path = _entry(cache, key)
    os.utime(path, ns=(mtime_ns, mtime_ns))


# ---------------------------------------------------------------------------
# cache_key
# ---------------------------------------------------------------------------


def test_cache_key_is_deterministic_hex() -> None:
    first = cache_key("tikz", "\\draw (0,0) -- (1,1);", "xelatex 1.0")
    second = cache_key("tikz", "\\draw (0,0) -- (1,1);", "xelatex 1.0")

    assert first == second
    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")


def test_cache_key_separator_prevents_ambiguity() -> None:
    assert cache_key("ab", "c") != cache_key("a", "bc")


def test_cache_key_distinguishes_empty_part_from_missing_part() -> None:
    assert cache_key("a") != cache_key("a", "")
    assert cache_key("", "") == cache_key("", "")


def test_cache_key_changes_with_toolchain_fingerprint() -> None:
    """RS-805: an upgraded xelatex or font must not hit the old render."""
    base = ("tikz", "\\draw (0,0) circle (1);", "default", "2100", "zh_CN")
    upgraded = base[:-2] + ("0.9999", "zh_CN")

    assert cache_key(*base) != cache_key(*upgraded)


def test_cache_key_empty_parts_is_stable() -> None:
    assert cache_key() == cache_key()
    assert len(cache_key()) == 64


def test_cache_key_rejects_non_str_part() -> None:
    with pytest.raises(TypeError):
        cache_key("tikz", b"bytes")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# DiskCache: hit / miss semantics
# ---------------------------------------------------------------------------


def test_put_then_get_returns_the_same_bytes(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    payload = b"<svg xmlns='http://www.w3.org/2000/svg'></svg>"

    cache.put(KEY, suffix=SUFFIX, data=payload)

    assert cache.get(KEY, suffix=SUFFIX) == payload
    assert _entry(cache).is_file()


def test_different_keys_do_not_hit(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    cache.put(KEY, suffix=SUFFIX, data=b"one")

    assert cache.get(OTHER_KEY, suffix=SUFFIX) is None


def test_different_suffixes_do_not_cross_talk(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    cache.put(KEY, suffix="svg", data=b"svg-bytes")

    assert cache.get(KEY, suffix="png") is None
    assert cache.get(KEY, suffix="svg") == b"svg-bytes"


def test_get_missing_root_returns_none(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "never-created")

    assert cache.get(KEY, suffix=SUFFIX) is None


@pytest.mark.parametrize(
    "key",
    [
        "",
        "0" * 63,
        "0" * 65,
        "A" * 64,
        "../" + "0" * 61,
        "0" * 32 + "-" + "0" * 31,
    ],
)
def test_get_with_invalid_key_is_a_miss(tmp_path: Path, key: str) -> None:
    cache = DiskCache(tmp_path)

    assert cache.get(key, suffix=SUFFIX) is None


@pytest.mark.parametrize("suffix", ["", "SVG", "svg!", "abcdefghi", "../svg", "s/g"])
def test_get_with_invalid_suffix_is_a_miss(tmp_path: Path, suffix: str) -> None:
    cache = DiskCache(tmp_path)
    cache.put(KEY, suffix=SUFFIX, data=b"payload")

    assert cache.get(KEY, suffix=suffix) is None


@pytest.mark.parametrize(
    "key", ["", "0" * 63, "A" * 64, "../" + "0" * 61]
)
def test_put_with_invalid_key_raises(tmp_path: Path, key: str) -> None:
    cache = DiskCache(tmp_path)

    with pytest.raises(ValueError):
        cache.put(key, suffix=SUFFIX, data=b"payload")


@pytest.mark.parametrize("suffix", ["", "SVG", "abcdefghi", "../svg", "s/g"])
def test_put_with_invalid_suffix_raises(tmp_path: Path, suffix: str) -> None:
    cache = DiskCache(tmp_path)

    with pytest.raises(ValueError):
        cache.put(KEY, suffix=suffix, data=b"payload")


def test_zero_length_file_is_a_miss_and_is_removed(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    cache.put(KEY, suffix=SUFFIX, data=b"")
    path = _entry(cache)
    assert path.is_file()

    assert cache.get(KEY, suffix=SUFFIX) is None
    assert not path.exists()


def test_get_swallows_read_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = DiskCache(tmp_path)
    cache.put(KEY, suffix=SUFFIX, data=b"payload")

    def boom(self: Path) -> bytes:
        raise OSError("simulated I/O error")

    monkeypatch.setattr(Path, "read_bytes", boom)

    assert cache.get(KEY, suffix=SUFFIX) is None


# ---------------------------------------------------------------------------
# DiskCache: atomic writes
# ---------------------------------------------------------------------------


def test_write_is_published_only_after_the_temporary_file_is_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = DiskCache(tmp_path)
    payload = b"x" * 10_000
    observed: dict[str, object] = {}
    real_replace = os.replace

    def spy_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
        observed["source"] = Path(src)
        observed["source_bytes"] = Path(src).read_bytes()
        real_replace(src, dst)

    monkeypatch.setattr(cache_module.os, "replace", spy_replace)

    cache.put(KEY, suffix=SUFFIX, data=payload)

    assert observed["source_bytes"] == payload
    assert not Path(observed["source"]).exists()  # type: ignore[arg-type]
    assert cache.get(KEY, suffix=SUFFIX) == payload


def test_put_leaves_no_temporary_files_behind(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)

    cache.put(KEY, suffix=SUFFIX, data=b"payload")

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        f"{KEY}.{SUFFIX}"
    ]


# ---------------------------------------------------------------------------
# DiskCache: capacity
# ---------------------------------------------------------------------------


def test_eviction_removes_least_recently_used_by_mtime(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, max_entries=2)
    _put_aged(cache, KEY, 1_000_000_000_000)
    _put_aged(cache, OTHER_KEY, 2_000_000_000_000)
    third = "2" * 64
    _put_aged(cache, third, 3_000_000_000_000)
    fourth = "3" * 64

    cache.put(fourth, suffix=SUFFIX, data=b"payload")

    survivors = sorted(path.name for path in tmp_path.iterdir())
    assert survivors == [f"{third}.{SUFFIX}", f"{fourth}.{SUFFIX}"]
    assert not _entry(cache).exists()


def test_max_entries_one_keeps_only_the_newest(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, max_entries=1)
    _put_aged(cache, KEY, 1_000_000_000_000)

    cache.put(OTHER_KEY, suffix=SUFFIX, data=b"newer")

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        f"{OTHER_KEY}.{SUFFIX}"
    ]


def test_eviction_ignores_foreign_files(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path, max_entries=1)
    foreign = tmp_path / "notes.txt"
    foreign.write_text("keep me")
    _put_aged(cache, KEY, 1_000_000_000_000)

    cache.put(OTHER_KEY, suffix=SUFFIX, data=b"newer")

    assert foreign.read_text() == "keep me"
    assert not _entry(cache).exists()


def test_get_refreshes_mtime_for_lru(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    cache.put(KEY, suffix=SUFFIX, data=b"payload")
    path = _entry(cache)
    os.utime(path, ns=(1_000_000_000_000, 1_000_000_000_000))

    assert cache.get(KEY, suffix=SUFFIX) == b"payload"

    assert path.stat().st_mtime_ns > 1_000_000_000_000


@pytest.mark.parametrize("max_entries", [0, -1, True])
def test_invalid_max_entries_is_rejected(
    tmp_path: Path, max_entries: int
) -> None:
    with pytest.raises(ValueError):
        DiskCache(tmp_path, max_entries=max_entries)


# ---------------------------------------------------------------------------
# DiskCache: failure degradation
# ---------------------------------------------------------------------------


def test_put_swallows_unavailable_root(tmp_path: Path) -> None:
    """A root that cannot be created (here: a regular file) must not raise."""
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("occupied")
    cache = DiskCache(blocker)

    cache.put(KEY, suffix=SUFFIX, data=b"payload")

    assert cache.get(KEY, suffix=SUFFIX) is None
    assert blocker.read_text() == "occupied"


def test_put_swallows_replace_failure_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = DiskCache(tmp_path)

    def boom(src: str, dst: str) -> None:
        raise OSError("simulated read-only directory")

    monkeypatch.setattr(cache_module.os, "replace", boom)

    cache.put(KEY, suffix=SUFFIX, data=b"payload")

    assert list(tmp_path.iterdir()) == []
    assert cache.get(KEY, suffix=SUFFIX) is None


# ---------------------------------------------------------------------------
# DiskCache: root resolution
# ---------------------------------------------------------------------------


def test_default_root_is_outside_the_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)

    cache = DiskCache()

    expected = Path(tempfile.gettempdir()) / DEFAULT_DIR_NAME
    assert cache.root == expected
    assert "packages" not in cache.root.parts
    assert cache.root != Path.cwd()
    assert Path.cwd() not in cache.root.parents


def test_env_var_selects_the_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = tmp_path / "from-env"
    monkeypatch.setenv(ENV_VAR, str(configured))

    cache = DiskCache()
    cache.put(KEY, suffix=SUFFIX, data=b"payload")

    assert cache.root == configured
    assert (configured / f"{KEY}.{SUFFIX}").read_bytes() == b"payload"


def test_explicit_root_wins_over_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "from-env"))
    explicit = tmp_path / "explicit"

    assert DiskCache(explicit).root == explicit


def test_stale_temp_files_are_swept(tmp_path: Path, monkeypatch) -> None:
    """A writer that dies before os.replace must not leak files forever."""
    root = tmp_path / "cache"
    root.mkdir()
    stale = root / f".{'a' * 64}.abc123{_tmp_suffix()}"
    fresh = root / f".{'b' * 64}.def456{_tmp_suffix()}"
    foreign = root / "notes.tmp"
    for path in (stale, fresh, foreign):
        path.write_bytes(b"x")
    old = time.time() - TEMP_FILE_MAX_AGE_SECONDS - 60
    os.utime(stale, (old, old))
    os.utime(foreign, (old, old))

    DiskCache(root).put(cache_key("k"), suffix="json", data=b"payload")

    assert not stale.exists(), "abandoned temp file must be swept"
    assert fresh.exists(), "a live writer's temp file must survive"
    assert foreign.exists(), "a user's file must never be deleted"


def test_sweep_never_deletes_a_non_cache_lookalike(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    root.mkdir()
    # Right shape but wrong prefix length / upper case / no leading dot.
    candidates = [f"{'a' * 63}.x.tmp", f".{'a' * 63}.x.tmp", f".{'A' * 64}.x.tmp"]
    for name in candidates:
        path = root / name
        path.write_bytes(b"x")
        old = time.time() - TEMP_FILE_MAX_AGE_SECONDS - 60
        os.utime(path, (old, old))

    DiskCache(root).put(cache_key("k"), suffix="json", data=b"payload")

    assert all((root / name).exists() for name in candidates)


# ---------------------------------------------------------------------------
# DiskCache: byte limit
# ---------------------------------------------------------------------------


def test_byte_limit_env_evicts_oldest_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(cache_module.MAX_BYTES_ENV_VAR, "10")
    cache = DiskCache(tmp_path)
    # 7 + 8 = 15 bytes > 10: the oldest entry must go, the newest must stay.
    _put_aged(cache, KEY, 1_000, data=b"payload")
    _put_aged(cache, OTHER_KEY, 2_000, data=b"12345678")

    assert not _entry(cache, KEY).exists()
    assert _entry(cache, OTHER_KEY).exists()


def test_default_byte_limit_applies_when_env_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(cache_module.MAX_BYTES_ENV_VAR, raising=False)
    # The real default is 256 MiB; shrink it so the test stays cheap while
    # still proving the *default* (not the env var) is what gets applied.
    monkeypatch.setattr(cache_module, "_DEFAULT_MAX_BYTES", 8)
    cache = DiskCache(tmp_path)
    _put_aged(cache, KEY, 1_000, data=b"payload")
    _put_aged(cache, OTHER_KEY, 2_000, data=b"12345678")

    assert not _entry(cache, KEY).exists()
    assert _entry(cache, OTHER_KEY).exists()


def test_zero_env_disables_the_byte_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(cache_module.MAX_BYTES_ENV_VAR, "0")
    cache = DiskCache(tmp_path)
    # 15 bytes would exceed the byte limits used by the eviction tests above.
    _put_aged(cache, KEY, 1_000, data=b"payload")
    _put_aged(cache, OTHER_KEY, 2_000, data=b"12345678")

    assert _entry(cache, KEY).exists()
    assert _entry(cache, OTHER_KEY).exists()


@pytest.mark.parametrize("raw", ["abc", "-5", "3.5"])
def test_invalid_env_falls_back_to_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    raw: str,
) -> None:
    monkeypatch.setenv(cache_module.MAX_BYTES_ENV_VAR, raw)
    monkeypatch.setattr(cache_module, "_DEFAULT_MAX_BYTES", 8)
    caplog.set_level(logging.WARNING, logger=cache_module.__name__)
    cache = DiskCache(tmp_path)
    _put_aged(cache, KEY, 1_000, data=b"payload")
    _put_aged(cache, OTHER_KEY, 2_000, data=b"12345678")

    # The default limit was applied (8 bytes), not "unlimited" and not a crash.
    assert not _entry(cache, KEY).exists()
    assert _entry(cache, OTHER_KEY).exists()

    warnings = [
        record
        for record in caplog.records
        if record.name == cache_module.__name__ and record.levelno == logging.WARNING
    ]
    assert warnings, "an unusable MAX_BYTES value must be reported"
    assert cache_module.MAX_BYTES_ENV_VAR in warnings[0].getMessage()


def test_total_exactly_at_limit_is_not_evicted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(cache_module.MAX_BYTES_ENV_VAR, "15")
    cache = DiskCache(tmp_path)
    _put_aged(cache, KEY, 1_000, data=b"payload")
    _put_aged(cache, OTHER_KEY, 2_000, data=b"12345678")

    assert _entry(cache, KEY).exists()
    assert _entry(cache, OTHER_KEY).exists()


def test_after_byte_eviction_hit_and_miss_behaviour_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(cache_module.MAX_BYTES_ENV_VAR, "8")
    cache = DiskCache(tmp_path)
    _put_aged(cache, KEY, 1_000, data=b"payload")
    _put_aged(cache, OTHER_KEY, 2_000, data=b"12345678")

    # The evicted entry is a plain miss, and a miss does not resurrect it.
    assert cache.get(KEY, suffix=SUFFIX) is None
    assert not _entry(cache, KEY).exists()
    # The survivor still returns the exact bytes and stays in place.
    assert cache.get(OTHER_KEY, suffix=SUFFIX) == b"12345678"
    assert _entry(cache, OTHER_KEY).exists()


def test_byte_limit_and_max_entries_evict_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(cache_module.MAX_BYTES_ENV_VAR, "8")
    newest = "2" * 64
    cache = DiskCache(tmp_path, max_entries=2)
    _put_aged(cache, KEY, 1_000, data=b"payload")
    _put_aged(cache, OTHER_KEY, 2_000, data=b"payload")
    _put_aged(cache, newest, 3_000, data=b"payload")

    # max_entries=2 removes the oldest entry, then the byte limit removes
    # the next-oldest one: only the newest entry survives both passes.
    assert not _entry(cache, KEY).exists()
    assert not _entry(cache, OTHER_KEY).exists()
    assert _entry(cache, newest).exists()
    assert cache.get(newest, suffix=SUFFIX) == b"payload"


def test_default_byte_limit_constant_is_256_mib() -> None:
    assert cache_module._DEFAULT_MAX_BYTES == 256 * 1024 * 1024


def test_byte_eviction_follows_recency_not_insertion_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(cache_module.MAX_BYTES_ENV_VAR, "14")
    newest = "2" * 64
    cache = DiskCache(tmp_path)
    _put_aged(cache, KEY, 1_000, data=b"payload")
    _put_aged(cache, OTHER_KEY, 2_000, data=b"payload")

    # A read refreshes ``KEY``'s mtime, so the *unread* entry is the LRU one
    # even though it was written later.
    assert cache.get(KEY, suffix=SUFFIX) == b"payload"
    _put_aged(cache, newest, 3_000, data=b"payload")

    assert _entry(cache, KEY).exists()
    assert not _entry(cache, OTHER_KEY).exists()
    assert _entry(cache, newest).exists()


def test_unlimited_bytes_still_enforces_max_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(cache_module.MAX_BYTES_ENV_VAR, "0")
    newest = "2" * 64
    cache = DiskCache(tmp_path, max_entries=2)
    _put_aged(cache, KEY, 1_000, data=b"payload")
    _put_aged(cache, OTHER_KEY, 2_000, data=b"payload")
    _put_aged(cache, newest, 3_000, data=b"payload")

    assert not _entry(cache, KEY).exists()
    assert _entry(cache, OTHER_KEY).exists()
    assert _entry(cache, newest).exists()


def test_foreign_files_do_not_count_towards_the_byte_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(cache_module.MAX_BYTES_ENV_VAR, "8")
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "someone-elses.bin").write_bytes(b"z" * 1024)
    cache = DiskCache(tmp_path)
    _put_aged(cache, KEY, 1_000, data=b"12345678")

    assert _entry(cache, KEY).exists()
    assert (tmp_path / "someone-elses.bin").exists()


def test_blob_larger_than_the_limit_is_evicted_immediately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Documented limitation: such a key is a permanent miss."""
    monkeypatch.setenv(cache_module.MAX_BYTES_ENV_VAR, "4")
    cache = DiskCache(tmp_path)
    # Not ``_put_aged``: the file is gone by the time ``put`` returns, so
    # there is nothing left to pin an mtime on.
    cache.put(KEY, suffix=SUFFIX, data=b"far more than four bytes")

    assert not _entry(cache, KEY).exists()
    assert cache.get(KEY, suffix=SUFFIX) is None


def test_unlimited_mode_never_reads_entry_sizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``0`` must skip the byte accounting, not merely exceed it."""
    monkeypatch.setenv(cache_module.MAX_BYTES_ENV_VAR, "0")
    cache = DiskCache(tmp_path)
    cache.put(KEY, suffix=SUFFIX, data=b"payload")
    original_stat = Path.stat

    def trap_size(self: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = original_stat(self, *args, **kwargs)

        class _Trap:
            def __getattr__(self, name: str):
                if name == "st_size":
                    raise AssertionError("unlimited mode must not sum sizes")
                return getattr(result, name)

        return _Trap()

    monkeypatch.setattr(Path, "stat", trap_size)
    cache.put(OTHER_KEY, suffix=SUFFIX, data=b"payload")

    assert cache.get(OTHER_KEY, suffix=SUFFIX) == b"payload"
