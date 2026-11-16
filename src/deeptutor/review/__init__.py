"""Review Studio: ReviewSpec, scheduling adapter, offline export.

The module is a thin layer on top of ``deeptutor.learning.scheduler``. It owns
no interval algorithm of its own; every due date is produced by the core
scheduler so a concept cannot have two different deadlines on two pages.
"""

from deeptutor.review.models import ReviewSpec, canonical_json, content_hash

__all__ = ("ReviewSpec", "canonical_json", "content_hash")
