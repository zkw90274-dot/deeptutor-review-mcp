from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("overlay", [False, True], ids=["spec", "overlay"])
def test_http_attempt_writes_real_mastery_and_replays_once(tmp_path: Path, overlay: bool) -> None:
    source = Path(__file__).resolve().parents[1] / "src" / "deeptutor"
    program = f"""
import deeptutor
deeptutor.__path__.insert(0, {str(source)!r})
from fastapi.testclient import TestClient
from deeptutor.learning.models import KnowledgePoint, LearningModule, LearningProgress
from deeptutor.learning.storage import LearningStore
from deeptutor.review.fixture import build_fixture_spec
from deeptutor.review.models import ReviewSpec
from deeptutor.review.service import ReviewService
from deeptutor.api.main import app
store = LearningStore()
store.save(LearningProgress(book_id='real-projection', name='真实写回测试', modules=[LearningModule(id='m', name='模块', order=1, knowledge_points=[KnowledgePoint(id='kp', name='串联谐振', type='concept', module_id='m')])]))
payload = build_fixture_spec().model_dump(mode='json')
payload['content_hash'] = ''
if not {overlay!r}:
    payload['mastery_path_id'] = 'real-projection'
    payload['concepts'][0]['knowledge_point_id'] = 'kp'
spec = ReviewSpec.model_validate(payload)
review = ReviewService()
review.put_spec(spec)
if {overlay!r}:
    import asyncio, json
    from deeptutor.review.agent_tools import ReviewLinkTool
    binding = asyncio.run(ReviewLinkTool().execute(action='bind', spec_id=spec.id, concept_id=spec.concepts[0].id, mastery_path_id='real-projection', knowledge_point_id='kp'))
    assert binding.success and json.loads(binding.content)['linked_concepts'] == 1, binding.content
    assert review.store.get_spec(spec.id).content_hash == spec.content_hash
client = TestClient(app)
attempt = {{'item_id': 'item-resonance-choice', 'rating': 'good', 'answer': 1, 'submission_id': 'real-once', 'session_id': 'real-session'}}
response = client.post('/api/review/' + spec.id + '/attempt', json=attempt)
assert response.status_code == 200, response.text
body = response.json()
assert body['mastery_projection']['status'] == 'applied', body
assert body['state_source'] == 'core', body
progress = store.load('real-projection')
assert len(progress.learning_evidence) == 1
assert progress.repetition_states['kp'].review_count == 1
replay = client.post('/api/review/' + spec.id + '/attempt', json=attempt)
assert replay.status_code == 200 and not replay.json()['created'], replay.text
assert len(store.load('real-projection').learning_evidence) == 1
assert store.load('real-projection').repetition_states['kp'].review_count == 1
print('REAL_CORE_PROJECTION_PASSED')
"""
    environment = {
        **os.environ, "DEEPTUTOR_HOME": str(tmp_path / "core"),
        "DEEPTUTOR_REVIEW_HOME": str(tmp_path / "review"), "AUTH_ENABLED": "false",
        "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
    }
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", program], env=environment,
        capture_output=True, encoding="utf-8", timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "REAL_CORE_PROJECTION_PASSED" in result.stdout
