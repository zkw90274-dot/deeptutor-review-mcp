from __future__ import annotations

import hashlib
import json
import runpy
import sys
from pathlib import Path

import pytest

from deeptutor.review import provenance
from deeptutor.review.models import ReviewSpec


def test_chapter_documents_preserve_audited_scope(tmp_path, monkeypatch) -> None:
    from deeptutor.review import generate

    raw = tmp_path / "course" / "raw"
    raw.mkdir(parents=True)
    (raw / "001_第2章_谐振.md").write_text("Resonance is at ５０ Hz.", encoding="utf-8")
    (raw / "002_notes__第2章_谐振.md").write_text("Secondary chapter copy.", encoding="utf-8")
    (raw / "003_第3章_滤波.md").write_text("Different chapter.", encoding="utf-8")
    monkeypatch.setattr(generate, "_knowledge_bases_root", lambda: tmp_path)
    assert provenance.chapter_documents("course", "第2章_谐振") == [
        ("001_第2章_谐振.md", "Resonance is at ５０ Hz."),
    ]


def test_subject_anchors_exact_term_weak_none() -> None:
    documents = [("chapter.md", "The resonance frequency is 50 Hz. The capacitor stores energy.")]
    assert provenance.match_anchor(documents, ["The resonance frequency is ５０ Hz"], [], [])['support_level'] == 'exact'
    assert provenance.match_anchor(documents, ["Resonance frequency measures 50 Hz"], ["resonance frequency"], ["50"])['support_level'] == 'term'
    assert provenance.match_anchor(documents, ["The capacitor loses energy"], [], [])['support_level'] == 'weak'
    assert provenance.match_anchor(documents, ["xyz"], [], [])['support_level'] == 'none'


def test_chinese_markup_and_offsets():
    raw = "# 测试标题\r\n\r\n**串联谐振时，电流与端电压同相。**\r\n"
    anchor = provenance.match_anchor([("chapter.md", raw)], ["串联谐振时, 电流与端电压同相"], [], [])
    assert anchor["support_level"] == "exact"
    assert anchor["support_reason"] == "normalized_statement"
    assert raw[anchor["char_start"]:anchor["char_end"]] == anchor["quote"]
    assert hashlib.sha256(anchor["quote"].encode()).hexdigest() == anchor["quote_hash"]


def test_paraphrase_with_equivalent_frequency():
    raw = r"振荡频率的测量值为 $１０００\,\mathrm{Hz}$。"
    anchor = provenance.match_anchor([("chapter.md", raw)], ["测得振荡频率为1kHz"], ["振荡频率"], [])
    assert anchor["support_level"] == "term"
    assert anchor["support_reason"] == "quantity_and_context"
    assert "1000 Hz" in anchor["matched_terms"]


def test_formula_wrappers_preserve_mathematical_identity():
    raw = r"$$\displaystyle Q=\dfrac{\omega_{0}L}{R}$$"
    anchor = provenance.match_anchor([("chapter.md", raw)], [r"\(Q = \frac{\omega_0 L}{R}\)"], ["品质因数"], [])
    assert anchor["support_level"] == "term"
    assert anchor["support_reason"] == "formula_answer"
    assert anchor["matched_terms"]


@pytest.mark.parametrize(("raw", "claim"), [
    ("# 串联谐振回路阻抗", "串联谐振回路阻抗"),
    ("这一点不正确。", "不正确"),
    ("串联谐振回路阻抗。", "串联谐振回路阻抗"),
    ("```text\n装置输出频率稳定不变。\n```", "装置输出频率稳定不变"),
    ("振荡频率为 1000 mHz。", "振荡频率为 1 kHz"),
    ("振荡频率为 -1000 Hz。", "振荡频率为 1 kHz"),
    ("振荡频率为 1e-3 Hz。", "振荡频率为 3 Hz"),
    ("振荡频率范围为 20～1000 Hz。", "振荡频率为 1 kHz"),
    ("振荡频率不是 1000 Hz。", "振荡频率为 1 kHz"),
    ("在反例中电压电流不同相。", "电压电流同相"),
    ("Bandwidth is unchanged. Inductor current at 1000 Hz is shown.", "Bandwidth is 1 kHz"),
    ("The source operates at 20 kHz.", "20 kHz"),
    (r"$Q=\frac{L}{R+C}$", r"$Q=\frac{L}{R}+C$"),
    (r"$Q=\frac{L}{R}$", r"$q=\frac{L}{R}$"),
    (r"| $Q=\frac{L}{R}$ | ❌ |", r"$Q=\frac{L}{R}$"),
])
def test_unsupported_evidence_cannot_be_strong(raw, claim):
    result = provenance.match_anchor([("chapter.md", raw)], [claim], ["串联谐振回路阻抗", "振荡频率"], ["1 kHz"])
    assert result["support_level"] in {"weak", "none"}
    assert result["support_reason"]
    assert len(result["matched_terms"]) <= 8


def test_chinese_rephrasing_without_verified_relation_is_weak():
    result = provenance.match_anchor(
        [("chapter.md", "并联谐振时感性支路和容性支路的电流相互抵消。")],
        ["并联谐振时，感性支路电流与容性支路电流互相抵消"], ["并联谐振"], [],
    )
    assert result["support_level"] == "weak"
    assert result["support_reason"] == "ngram_overlap"


def test_audit_script_reads_without_refreshing(service, fixture_spec, tmp_path, monkeypatch, capsys):
    service.put_spec(fixture_spec)
    script = Path(__file__).resolve().parents[1] / "scripts/audit_provenance.py"
    database = service.store.db_path
    before = database.read_bytes()
    monkeypatch.setattr(sys, "argv", [str(script), "--home", str(database.parent),
                                      "--kb-root", str(tmp_path / "kb"), "--spec", fixture_spec.id])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(script), run_name="__main__")
    assert result.value.code == 0
    audit = json.loads(capsys.readouterr().out)
    assert audit["content_hash"] == fixture_spec.content_hash
    assert audit["counts"] == {"exact": 0, "term": 0, "weak": 0, "none": len(fixture_spec.items)}
    assert all(item["support_reason"] == "chapter_unavailable" for item in audit["items"])
    assert database.read_bytes() == before


def test_raw_chapter_reads_heading_matched_document_without_changing_fingerprint_scope(tmp_path, monkeypatch):
    from deeptutor.review import generate

    raw = tmp_path / "course" / "raw"
    raw.mkdir(parents=True)
    body = "# 第2章 谐振\r\n正文内容\r\n"
    (raw / "supplement.md").write_bytes(body.encode())
    (raw / "unrelated.md").write_text("# 第3章 谐振\n", encoding="utf-8")
    monkeypatch.setattr(generate, "_knowledge_bases_root", lambda: tmp_path)
    assert provenance.chapter_paths("course", "第2章_谐振") == []
    assert provenance.chapter_documents("course", "第2章_谐振") == [("supplement.md", body)]


def test_quality_state_legacy_draft_verified(service, fixture_spec, monkeypatch) -> None:
    payload = fixture_spec.model_dump(mode="json")
    payload["content_hash"] = ""
    for source in payload["sources"].values():
        source.update(kind="doc", kb="hf", ref="第2章")
    for item, explanation in zip(payload["items"], (
        "串联回路的感抗与容抗相互抵消。", "回路谐振频率约为 1.592 MHz。", "品质因数反映储能和损耗的比值。",
    ), strict=True):
        item["explanation"] += "\n" + explanation
    fixture_spec = ReviewSpec.model_validate(payload)
    monkeypatch.setattr(provenance, 'chapter_documents', lambda *_: [])
    service.put_spec(fixture_spec)
    assert service.quality(fixture_spec.id)['state'] == 'draft'
    assert not service.today(fixture_spec.id)['items']
    assert service.today(fixture_spec.id, include_draft=True)['items']
    monkeypatch.setattr(provenance, 'chapter_documents', lambda *_: [
        ('chapter.md', '\n'.join(item.explanation for item in fixture_spec.items))
    ])
    report = service.quality(fixture_spec.id, refresh=True)
    assert report["state"] == "verified", (report["support"], report["answer_leaks"])


def test_legacy_packs_still_schedule(service, fixture_spec, monkeypatch) -> None:
    from deeptutor.review.storage import ReviewStore

    service.store.put_spec(fixture_spec)
    with service.store._transaction() as conn:
        conn.execute("DELETE FROM review_schema_migrations")
        conn.execute("INSERT INTO review_schema_migrations VALUES (5, 0)")
        conn.execute("DROP TABLE review_quality")
    service.store = ReviewStore(service.store.home)
    before = service.store.quality_record(fixture_spec.id)
    assert before["state"] == "legacy" and "support" not in before
    assert service.today(fixture_spec.id)["quality_warning"]
    assert service.today(fixture_spec.id)["items"]
    assert "support" not in service.store.quality_record(fixture_spec.id)
    assert service.quality(fixture_spec.id)["state"] == "legacy"
    assert service.quality(fixture_spec.id, refresh=True)["state"] == "draft"


def test_export_renders_snippet_and_watermark(service, fixture_spec) -> None:
    service.put_spec(fixture_spec)
    service.store.save_quality(fixture_spec, {"state": "draft"}, {
        fixture_spec.items[0].id: {"chapter": "Chapter A", "quote": "Sample source quote"},
    })
    page = service.export(fixture_spec.id)
    assert "未人工复核" in page
    assert "Sample source quote" in page and 'el("blockquote", "", anchor.quote)' in page
