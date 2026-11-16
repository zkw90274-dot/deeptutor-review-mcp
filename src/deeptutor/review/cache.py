"""Deterministic on-disk cache for rendered figures (RS-805).

Rendering a TikZ figure costs a ``xelatex`` compile plus a ``pdftocairo``
conversion, so the same source must not be rendered twice. The cache is
deliberately boring: a flat directory, one file per ``(key, suffix)`` pair, no
database, no lock files.

Four properties matter more than throughput:

* **Content addressed.** :func:`cache_key` derives the key from caller-supplied
  parts (provider, source, tokens, style, toolchain, locale). Time, paths and
  process ids never enter the key, so a warm cache is reproducible across
  machines. The toolchain fingerprint is a caller-supplied part on purpose:
  after a ``xelatex`` upgrade the old SVG must not be reused. Fonts are *not*
  fingerprinted here — the cache cannot see them, so a font file replaced in
  place at the same tool version yields the same key (see the phase document's
  known limitations). Deployments that reinstall fonts should bump the version
  part or clear the cache.
* **Never break the export.** A cache is an optimisation. Read failures are a
  miss, write failures are logged and ignored, eviction failures are ignored.
  Only programmer errors (a malformed key or suffix passed to ``put``) raise.
* **Torn files are impossible.** Data is written to a unique temporary file in
  the cache directory and published with :func:`os.replace`, which is atomic on
  Windows and POSIX. Concurrent writers of the same key may overwrite each
  other, but a reader only ever sees a complete file. A process that dies
  between those two steps leaves a hidden ``.tmp`` file; :meth:`DiskCache.put`
  sweeps stale ones so a crash loop cannot accumulate them forever.
* **Bounded storage.** After every write, eviction removes
  least-recently-used entries until both the entry count (``max_entries``)
  and the total size of content files fit their limits. The byte limit
  defaults to 256 MiB and can be overridden with
  ``DEEPTUTOR_REVIEW_CACHE_MAX_BYTES`` (``0`` disables it). Only content
  files matching the entry pattern count towards the limits; temporary
  files are excluded (they are swept once an hour old) and so are foreign
  files in a shared directory.

Known limitations of the byte bound:

* It is enforced on writes only. A directory grown by other means (another
  tool, a restored backup) is trimmed on the next ``put``, not on ``get``.
* A single blob larger than the whole limit cannot be retained: it is
  evicted by the very write that stored it, so that key is a permanent miss
  until the limit is raised or disabled.
* Size is measured on the files themselves, so filesystem block overhead and
  any future compression are not accounted for.

The default directory is ``$DEEPTUTOR_REVIEW_CACHE`` or, when unset,
``<system temp>/deeptutor-review-cache``. The working tree is never used.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
import time
from pathlib import Path

_LOGGER = logging.getLogger(__name__)

#: Field separator for :func:`cache_key`. ``\x1f`` (ASCII unit separator) is
#: illegal in the textual parts a caller passes and cannot appear in a hex
#: digest either, so ``("ab", "c")`` and ``("a", "bc")`` hash differently. A
#: trailing separator is appended after every part, so ``("a",)`` and
#: ``("a", "")`` are distinct as well.
_SEPARATOR = "\x1f"

#: A key is a lowercase SHA-256 hex digest. It becomes part of a filename, so
#: anything else (path separators, ``..``, uppercase) is rejected.
_KEY_PATTERN = re.compile(r"\A[0-9a-f]{64}\Z")

#: Suffixes are part of the filename too. The whitelist keeps ``../x`` and
#: drive letters out of the path; real callers use ``svg``, ``png`` or ``pdf``.
_SUFFIX_PATTERN = re.compile(r"\A[a-z0-9]{1,8}\Z")

#: Files this module owns. Only these are candidates for eviction, so a user
#: pointing ``DEEPTUTOR_REVIEW_CACHE`` at a shared directory cannot lose data.
_ENTRY_PATTERN = re.compile(r"\A[0-9a-f]{64}\.[a-z0-9]{1,8}\Z")

ENV_VAR = "DEEPTUTOR_REVIEW_CACHE"
DEFAULT_DIR_NAME = "deeptutor-review-cache"

#: Environment variable overriding the byte limit. ``0`` disables the limit;
#: anything that is not a non-negative integer falls back to the default.
MAX_BYTES_ENV_VAR = "DEEPTUTOR_REVIEW_CACHE_MAX_BYTES"

#: Default cap on the total size of content files, in bytes.
_DEFAULT_MAX_BYTES = 256 * 1024 * 1024

#: Temporary files are hidden (leading dot) and therefore never match
#: :data:`_ENTRY_PATTERN`; the suffix is chosen to make leaks obvious.
_TMP_SUFFIX = ".tmp"

#: What :meth:`DiskCache._sweep_temp_files` may delete: exactly the shape
#: :func:`tempfile.mkstemp` produces for this module, never a user's file.
_TEMP_PATTERN = re.compile(
    r"\A\.[0-9a-f]{64}\.[^\\/]*" + re.escape(_TMP_SUFFIX) + r"\Z"
)

#: A temp file older than this is abandoned: a render that is still running
#: holds it for seconds at most, so an hour is far beyond any live writer.
TEMP_FILE_MAX_AGE_SECONDS = 3600.0


def cache_key(*parts: str) -> str:
    """Return a stable 64-character hex key for ``parts``.

    Parts are UTF-8 encoded, joined with :data:`_SEPARATOR`, and hashed with
    SHA-256. The function is pure: the same arguments always produce the same
    key, and no part may be omitted without changing it.
    """
    digest = hashlib.sha256()
    for index, part in enumerate(parts):
        if not isinstance(part, str):
            raise TypeError(
                f"cache_key part {index} must be str, got {type(part).__name__}"
            )
        digest.update(part.encode("utf-8"))
        digest.update(_SEPARATOR.encode("ascii"))
    return digest.hexdigest()


def _is_valid_key(key: object) -> bool:
    return isinstance(key, str) and _KEY_PATTERN.match(key) is not None


def _is_valid_suffix(suffix: object) -> bool:
    return isinstance(suffix, str) and _SUFFIX_PATTERN.match(suffix) is not None


class DiskCache:
    """A size-bounded, content-addressed blob cache.

    ``root`` is created lazily on the first :meth:`put`; constructing a cache
    for a directory that does not exist yet is not an error. When ``root`` is
    ``None`` the ``DEEPTUTOR_REVIEW_CACHE`` environment variable is consulted
    and otherwise the system temporary directory is used.

    Storage is bounded two ways. ``max_entries`` caps the number of cached
    blobs; the byte limit caps their total size (default 256 MiB, override
    with ``DEEPTUTOR_REVIEW_CACHE_MAX_BYTES``, ``0`` = unlimited). Both are
    enforced after each write, evicting least-recently-used entries first.
    """

    def __init__(
        self, root: str | Path | None = None, *, max_entries: int = 256
    ) -> None:
        if (
            not isinstance(max_entries, int)
            or isinstance(max_entries, bool)
            or max_entries < 1
        ):
            raise ValueError("max_entries must be an integer >= 1")
        self._max_entries = max_entries
        self._max_bytes = self._resolve_max_bytes()
        self._root = self._resolve_root(root)

    @staticmethod
    def _resolve_root(root: str | Path | None) -> Path:
        if root is not None:
            return Path(root).expanduser()
        configured = os.environ.get(ENV_VAR, "").strip()
        if configured:
            return Path(configured).expanduser()
        return Path(tempfile.gettempdir()) / DEFAULT_DIR_NAME

    @staticmethod
    def _resolve_max_bytes() -> int:
        """Return the byte limit from the environment, or the default.

        ``0`` disables the limit. A negative or non-integer value is a
        configuration mistake, not a crash: it falls back to the default
        with a warning, mirroring how the read path never raises.
        """
        raw = os.environ.get(MAX_BYTES_ENV_VAR, "").strip()
        if not raw:
            return _DEFAULT_MAX_BYTES
        try:
            value = int(raw)
        except ValueError:
            _LOGGER.warning(
                "ignoring non-integer %s=%r; using default byte limit",
                MAX_BYTES_ENV_VAR,
                raw,
            )
            return _DEFAULT_MAX_BYTES
        if value < 0:
            _LOGGER.warning(
                "ignoring negative %s=%r; using default byte limit",
                MAX_BYTES_ENV_VAR,
                raw,
            )
            return _DEFAULT_MAX_BYTES
        return value

    @property
    def root(self) -> Path:
        """Directory backing this cache (not necessarily created yet)."""
        return self._root

    def get(self, key: str, *, suffix: str) -> bytes | None:
        """Return the cached blob, or ``None`` on any kind of miss.

        A malformed ``key`` or ``suffix`` is treated as a miss rather than an
        error: the read path runs inside export and must never fail. Empty
        files (a crashed or externally truncated writer) are deleted and
        reported as a miss. A successful hit refreshes the file's mtime, which
        is what makes eviction least-recently-*used* rather than FIFO.
        """
        if not _is_valid_key(key) or not _is_valid_suffix(suffix):
            return None
        target = self._root / f"{key}.{suffix}"
        try:
            data = target.read_bytes()
        except OSError:
            return None
        if not data:
            self._discard(target)
            return None
        try:
            os.utime(target, None)
        except OSError:
            pass
        return data

    def put(self, key: str, *, suffix: str, data: bytes) -> None:
        """Store ``data`` under ``key``/``suffix``, swallowing disk errors.

        A malformed ``key`` or ``suffix`` raises :class:`ValueError`: that is a
        caller bug, not a cache failure. Everything that can go wrong with the
        filesystem (missing permissions, a read-only directory, a full disk,
        Windows refusing to replace a file another process has open) is logged
        at debug level and ignored, because losing a cache write only costs one
        re-render.
        """
        if not _is_valid_key(key):
            raise ValueError("cache key must be 64 lowercase hex characters")
        if not _is_valid_suffix(suffix):
            raise ValueError("cache suffix must match [a-z0-9]{1,8}")
        payload = bytes(data)

        try:
            self._root.mkdir(parents=True, exist_ok=True)
            handle, tmp_name = tempfile.mkstemp(
                prefix=f".{key}.", suffix=_TMP_SUFFIX, dir=self._root
            )
        except OSError as exc:
            _LOGGER.debug("render cache unavailable at %s: %s", self._root, exc)
            return

        tmp_path = Path(tmp_name)
        try:
            stream = os.fdopen(handle, "wb")
        except OSError as exc:
            # ``fdopen`` only fails on a bad descriptor; nothing was written.
            _LOGGER.debug("render cache write failed for %s.%s: %s", key, suffix, exc)
            try:
                os.close(handle)
            except OSError:
                pass
            self._discard(tmp_path)
            return

        try:
            with stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp_path, self._root / f"{key}.{suffix}")
        except OSError as exc:
            _LOGGER.debug(
                "render cache write failed for %s.%s: %s", key, suffix, exc
            )
            self._discard(tmp_path)
            return

        self._evict()
        self._sweep_temp_files()

    def _discard(self, path: Path) -> None:
        try:
            path.unlink()
        except OSError:
            pass

    def _sweep_temp_files(self) -> None:
        """Delete leftovers from a writer that died before ``os.replace``.

        Temporary files are hidden and therefore invisible to eviction, so a
        process that keeps crashing mid-write would grow the directory without
        bound. Only files this module could have created are touched
        (``.<64 hex>.<name>.tmp``) and only once they are clearly abandoned.
        Failures are ignored: a stale temp file is untidy, not a defect.
        """
        try:
            cutoff = time.time() - TEMP_FILE_MAX_AGE_SECONDS
            for child in self._root.iterdir():
                if _TEMP_PATTERN.match(child.name) is None:
                    continue
                try:
                    if child.stat().st_mtime < cutoff:
                        self._discard(child)
                except OSError:
                    continue
        except OSError as exc:
            _LOGGER.debug("render cache sweep failed for %s: %s", self._root, exc)

    def _evict(self) -> None:
        """Delete least-recently-used entries until both limits hold.

        Only files matching :data:`_ENTRY_PATTERN` are considered, so foreign
        files in a shared directory are left alone. Entries beyond
        ``max_entries`` go first, then the oldest ones until the total content
        size fits. When the byte limit is disabled
        (``DEEPTUTOR_REVIEW_CACHE_MAX_BYTES=0``) no size is recorded, no sum is
        taken and the count branch returns on its own, so that path behaves
        exactly as it did before the limit existed. Every failure is ignored: a
        full cache is still a working cache.
        """
        try:
            entries: list[tuple[int, str, Path, int]] = []
            measure = self._max_bytes > 0
            for child in self._root.iterdir():
                if _ENTRY_PATTERN.match(child.name) is None:
                    continue
                try:
                    stat = child.stat()
                except OSError:
                    continue
                entries.append(
                    (
                        stat.st_mtime_ns,
                        child.name,
                        child,
                        stat.st_size if measure else 0,
                    )
                )

            excess = len(entries) - self._max_entries
            if not measure:
                # Unlimited: exactly the pre-byte-limit behaviour, so a cache
                # that opts out pays nothing for the accounting.
                if excess <= 0:
                    return
                entries.sort(key=lambda entry: (entry[0], entry[1]))
                for _, _, path, _ in entries[:excess]:
                    self._discard(path)
                return

            total_bytes = sum(entry[3] for entry in entries)
            if excess <= 0 and total_bytes <= self._max_bytes:
                return

            entries.sort(key=lambda entry: (entry[0], entry[1]))
            if excess > 0:
                for _, _, path, size in entries[:excess]:
                    self._discard(path)
                    total_bytes -= size
                del entries[:excess]

            for _, _, path, size in entries:
                if total_bytes <= self._max_bytes:
                    break
                self._discard(path)
                total_bytes -= size
        except OSError as exc:
            _LOGGER.debug(
                "render cache eviction failed for %s: %s", self._root, exc
            )
