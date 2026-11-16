"""Deterministic QA gates for ReviewSpec (RS-501/502/504/506/507).

Gates only check properties a machine can prove. They do not judge whether a
sentence is pedagogically correct; that remains a human/LLM review concern.
A failed blocking gate prevents the spec from being stored or exported.
"""

from __future__ import annotations

import re
import math
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

from deeptutor.review.models import GateResult, QualityReport, ReviewSpec

_MATH_PATTERN = re.compile(
    r"\$\$(?P<double>.+?)\$\$|\$(?P<single>.+?)\$|\\\((?P<paren>.+?)\\\)|\\\[(?P<bracket>.+?)\\\]",
    re.DOTALL,
)
_RAW_COMPARISON = re.compile(r"(?<!\\)[<>]")
_RAW_PIPE = re.compile(r"(?<!\\)\|")
_UNSAFE_PATTERNS = (
    re.compile(r"<\s*script\b", re.IGNORECASE),
    re.compile(r"<\s*/\s*script\b", re.IGNORECASE),
    re.compile(r"<\s*iframe\b", re.IGNORECASE),
    re.compile(r"<\s*object\b", re.IGNORECASE),
    re.compile(r"<\s*embed\b", re.IGNORECASE),
    re.compile(r"\bon[a-z]+\s*=", re.IGNORECASE),
    re.compile(r"javascript\s*:", re.IGNORECASE),
    re.compile(r"data\s*:\s*text/html", re.IGNORECASE),
)
_EXTERNAL_URL = re.compile(r"(?:https?:)?//", re.IGNORECASE)

#: Bigram coverage a short answer needs in the explanation before G4 accepts it.
#: Generated answers paraphrase instead of quoting, so an exact substring is
#: too strict; the offline audit imports this constant to stay in step with
#: the gate it mirrors.
ANSWER_COVERAGE_MIN = 0.6


def _strings(value: Any) -> Iterable[tuple[str, str]]:
    """Yield ``(path, string)`` for every string in a dumped model."""
    if isinstance(value, str):
        yield "", value
    elif isinstance(value, dict):
        for key, child in value.items():
            for path, text in _strings(child):
                yield (f"{key}.{path}" if path else str(key)), text
    elif isinstance(value, list):
        for index, child in enumerate(value):
            for path, text in _strings(child):
                yield (f"[{index}].{path}" if path else f"[{index}]"), text


def _math_segments(text: str) -> Iterable[str]:
    for match in _MATH_PATTERN.finditer(text):
        yield next(group for group in match.groups() if group is not None)


_FENCE_LINE = re.compile(r"^\s*(?:```+|~~~+)")


def _strip_fenced_code(text: str) -> str:
    """Drop fenced code before scanning for math.

    The export renders a fenced block as plain monospace text, so a literal
    ``$`` inside assembly or pseudo-code is not a math delimiter and must not
    make an otherwise valid pack fail G3.
    """
    lines: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if _FENCE_LINE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append(line)
    return "\n".join(lines)


def _gate_schema(spec: ReviewSpec) -> GateResult:
    if spec.schema_version != "1.0":
        return GateResult(gate="G1-schema", passed=False, message="unknown schema major")
    return GateResult(
        gate="G1-schema",
        passed=True,
        message="schema 1.0 validated",
        details=[f"items={len(spec.items)} concepts={len(spec.concepts)}"],
    )


def _gate_sources(spec: ReviewSpec) -> GateResult:
    details: list[str] = []
    for concept in spec.concepts:
        if concept.coverage == "kb" and not concept.source_ids:
            details.append(f"concept {concept.id}: kb coverage without source")
        for formula in concept.formulas:
            if not formula.source_ids and not concept.source_ids:
                details.append(f"concept {concept.id}: formula has no source")
    for item in spec.items:
        if not item.source_ids:
            details.append(f"item {item.id}: no source")
        missing = [source_id for source_id in item.source_ids if source_id not in spec.sources]
        for source_id in missing:
            details.append(f"item {item.id}: missing source {source_id}")
    return GateResult(
        gate="G2-sources",
        passed=not details,
        message="every concept/formula/item is anchored" if not details else "source coverage incomplete",
        details=details[:20],
    )


def _gate_math(spec: ReviewSpec) -> GateResult:
    details: list[str] = []
    for path, text in _strings(spec.model_dump(mode="json", exclude={"quality"})):
        for segment in _math_segments(_strip_fenced_code(text)):
            if _RAW_COMPARISON.search(segment):
                details.append(f"{path}: raw < or > inside math (use \\lt / \\gt)")
            if _RAW_PIPE.search(segment):
                details.append(f"{path}: raw | inside math (use \\vert{{}})")
    # Deduplicate while preserving order; one bad formula should not flood logs.
    unique = list(dict.fromkeys(details))
    return GateResult(
        gate="G3-math-escape",
        passed=not unique,
        message="math delimiters are safe" if not unique else "unsafe math delimiters",
        details=unique[:20],
    )


def _gate_math_balance(spec: ReviewSpec) -> GateResult:
    """Every math delimiter must close.

    An odd ``$`` count is not a syntax error MathJax reports: it silently
    typesets the pairs and leaves the stray ``$`` as literal text, so the page
    ends up showing raw LaTeX. Checking the delimiters here keeps that class
    out of stored specs; ``\\(``/``\\[`` are counted for the same reason.
    """
    details: list[str] = []
    dumped = spec.model_dump(mode="json", exclude={"quality"})
    for path, text in _strings(dumped):
        body = _strip_fenced_code(text)
        if body.count("$") % 2:
            details.append(f"{path}: unbalanced $ delimiters")
        if body.count("\\(") != body.count("\\)"):
            details.append(f"{path}: unbalanced \\( \\) delimiters")
        if body.count("\\[") != body.count("\\]"):
            details.append(f"{path}: unbalanced \\[ \\] delimiters")
    unique = list(dict.fromkeys(details))
    return GateResult(
        gate="G10-math-balance",
        passed=not unique,
        message="math delimiters are balanced" if not unique else "unbalanced math delimiters",
        details=unique[:20],
    )


def _text_bigrams(text: str) -> set[str]:
    normalized = re.sub(r"[^0-9A-Za-z一-鿿]+", "", text)
    if len(normalized) < 2:
        return {normalized} if normalized else set()
    return {normalized[index : index + 2] for index in range(len(normalized) - 1)}


# ``_text_bigrams`` strips punctuation, so ``AF<1`` and ``AF>1`` used to look
# identical to G4 (coverage 1.0). The signature check below keeps the
# comparison operators: plain text, ``\lt{}``/``\gt{}`` macros and the unicode
# forms ``≤``/``≥`` all normalize to the same tokens. ``=`` is deliberately not
# a comparison token here: plain equations must keep working.
_COMPARISON_FORMS = (
    (r"\\leq(?![A-Za-z])", "<="),
    (r"\\geq(?![A-Za-z])", ">="),
    (r"\\le(?![A-Za-z])", "<="),
    (r"\\ge(?![A-Za-z])", ">="),
    (r"\\lt(?![A-Za-z])", "<"),
    (r"\\gt(?![A-Za-z])", ">"),
    ("≤", "<="),
    ("≥", ">="),
)
_COMPARISON_TOKEN = re.compile(r"<=|>=|<|>")

# Negation scanning: the negator must sit in the same clause as the answer with
# at most 12 other characters in between ("不会因为负载变化而增大"). Compound
# words that merely *start* with 不 ("不断增大", "不同") are not negations.
_CLAUSE_BOUNDARY = re.compile(r"[，。；;、!?！？：:,\n]")
_DOUBLE_NEGATION = re.compile(r"不是不|并非不|不能不|不得不|不可不|未尝不|无不")
_NEGATOR = (
    r"(?:不可以|不会|不能|无法|不可|不应|并非|不是|没有|没能|未曾|不曾|没|未"
    r"|不(?!断|同|仅|管|少|久|论|但|过|得|妨|光|只))"
)
_NEGATED_TAIL = re.compile(_NEGATOR + r"[^，。；;、!?！？：:,\n]{0,12}$")


def _comparison_signature(text: str) -> tuple[str, ...]:
    normalized = text
    for pattern, replacement in _COMPARISON_FORMS:
        normalized = re.sub(pattern, replacement, normalized)
    return tuple(sorted(_COMPARISON_TOKEN.findall(normalized)))


def _negated_prefix(compact: str, start: int) -> bool:
    """Whether the clause right before ``start`` negates what follows it."""
    window = compact[max(0, start - 64) : start]
    prefix = _CLAUSE_BOUNDARY.split(window)[-1]
    prefix = _DOUBLE_NEGATION.sub("肯定", prefix)
    return bool(_NEGATED_TAIL.search(prefix))


_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def answer_coverage(explanation: str, answer: str) -> float:
    """Share of the answer's character bigrams that the explanation covers.

    Generated short answers often paraphrase the reference answer instead of
    copying it. Requiring an exact substring forces the generator to append
    the whole answer again, so G4 accepts a high-coverage paraphrase instead.
    """
    explanation = unicodedata.normalize("NFKC", explanation)
    answer = unicodedata.normalize("NFKC", answer).strip()
    compact = re.sub(r"\s+", "", explanation)
    needle = re.sub(r"\s+", "", answer)
    if needle:
        occurrences = [match.start() for match in re.finditer(re.escape(needle), compact)]
        if occurrences:
            if all(_negated_prefix(compact, start) for start in occurrences):
                return 0.0
            # A verbatim hit is the strongest evidence there is, so it must not
            # be vetoed by the operator signature: explanations routinely
            # compare other quantities ("Q>10 时 ... <answer> ... Q<1"). Numeric
            # and single-character answers keep the strict boundary checks
            # below, where a bare digit or letter may sit inside another token.
            if len(needle) > 1 and not _NUMBER.fullmatch(needle):
                return 1.0
    if needle:
        answer_operators = _comparison_signature(answer)
        if answer_operators and _comparison_signature(explanation) != answer_operators:
            return 0.0
    if _NUMBER.fullmatch(needle):
        return float(_mentions_number(explanation, needle))
    if len(needle) == 1:
        if needle.isascii() and needle.isalnum():
            return float(bool(re.search(rf"(?<![A-Za-z0-9.]){re.escape(needle)}(?![A-Za-z0-9.]|[eE][-+]?\d)", compact)))
        return float(needle in compact)
    expected = _text_bigrams(answer)
    if not expected:
        return float(bool(needle) and needle in compact)
    return len(expected & _text_bigrams(explanation)) / len(expected)


# ``2×I_c1m×R_p`` (explicit), ``2 I_c1m R_p`` and ``2I_{c1m}R_p`` (implicit
# multiplication) all state twice the amplitude. ``_normalized_math_text`` has
# already removed the braces and the ``\mathrm`` wrappers by this point.
_DOUBLED_PRODUCT = r"2\s*[×*·]?\s*I_?c1m\s*[×*·]?\s*R_?p"


def _assignment(text: str, names: str) -> float | None:
    pattern = re.compile(rf"(?<![A-Za-z0-9])(?:{names})\s*=", re.IGNORECASE)
    for index, match in enumerate(pattern.finditer(text)):
        if index >= 32:
            break
        # The value may be the last standalone number of a chained equation:
        # ``I_c1m=α1 i_Cmax=0.5 A`` or ``P_o=...=6.25 W``. Numbers inside
        # identifiers (``α1``, ``I_c1m``, exponent ``^2``) are not values.
        clause = re.split(
            r"[，。；、,;\n]|\b(?:and|but)\b",
            text[match.end() :match.end() + 160], maxsplit=1, flags=re.IGNORECASE,
        )[0]
        number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
        if not re.match(rf"\s*{number}", clause) or (
            "=" in clause and not re.search(r"[A-Za-zξα][A-Za-z0-9_]*\s*=", clause)
        ):
            clause = clause.rsplit("=", 1)[-1]
        parsed = re.match(rf"\s*({number})\s*([pnumkMμµ]?)(V|A|Ω|ohm)?(?![A-Za-z0-9_.])", clause)
        if parsed:
            scale = {"": 1.0, "p": 1e-12, "n": 1e-9, "u": 1e-6, "μ": 1e-6, "µ": 1e-6, "m": 1e-3, "k": 1e3, "M": 1e6}
            value = float(parsed.group(1)) * (scale[parsed.group(2)] if parsed.group(3) else 1.0)
            if math.isfinite(value):
                return value
    return None


def _normalized_math_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("$", " ")
    for _ in range(8):
        unwrapped = re.sub(r"\\(?:mathrm|text|textrm|operatorname)\s*\{((?:[^{}]|\{[^{}]*\})*)\}", r"\1", text)
        if unwrapped == text:
            break
        text = unwrapped
    text = re.sub(r"\\[A-Za-z]+", lambda match: match.group(0)[1:], text)
    return text.replace("{", "").replace("}", "").replace("\\", "")


def parameter_infeasibility(item: Any) -> str | None:
    """A cheap feasibility check for resonant power-amplifier parameters.

    It only fires when the text states the relevant quantities, so unrelated
    items are untouched. A voltage swing above ``V_CC`` (or ``ξ > 1``) is
    impossible regardless of the conduction angle.
    """
    answer = getattr(item, "answer", None)
    stem = str(getattr(item, "stem", ""))
    answer_text = str(getattr(answer, "text", "") or "")
    options = list(getattr(item, "options", None) or [])
    choice_index = getattr(answer, "choice_index", None)
    if isinstance(choice_index, int) and 0 <= choice_index < len(options):
        answer_text = str(getattr(options[choice_index], "text", ""))
    if re.search(
        r"是否(?:可能|合理|可行)|能否|会怎样|会发生什么|(?:can|could|is|are)\b.*\b(?:feasible|possible)",
        stem, re.IGNORECASE,
    ) and re.search(
        r"^\s*(?:(?:这|该|上述|给定)(?:组)?(?:参数|条件|假设|工况)(?:是|为)?\s*)?"
        r"(?:不可能(?:成立)?|不可行|不合理|不成立|无法实现|否|不能)(?=[，。；：:,.!！\s]|$)"
        r"|^\s*(?:(?:these|the|given)\s+(?:parameters|conditions|assumptions)\s+(?:are|is)\s+)?"
        r"(?:not\s+(?:physically\s+)?(?:feasible|possible)|impossible|infeasible)\b",
        answer_text, re.IGNORECASE,
    ):
        return None
    text = _normalized_math_text(
        "\n".join(
            [
                str(getattr(item, "stem", "")),
                str(getattr(item, "explanation", "")),
                str(getattr(answer, "text", "") or ""),
            ]
        )
    )
    rf_context = bool(re.search(r"谐振功|谐振功率|丙类功|resonant.*amplif", text, re.IGNORECASE))
    vcc = _assignment(text, r"V_?CC|VCC")
    if vcc is None and rf_context:
        vcc = _assignment(text, r"E_?c")
    rp = _assignment(text, r"R_?p|Rp")
    ic1m = _assignment(text, r"I_?c1m|Ic1m")
    xi = _assignment(text, r"xi|ξ")
    icmax = _assignment(text, r"i_?Cmax|I_?Cmax|i_?cmax|I_?cmax")
    alpha1 = _assignment(text, r"alpha_?1|α_?1|α1")

    # ``V_cm=1.2×V_CC``: a scalar multiple of the supply states the swing
    # directly. It must win over ``_assignment``, which would otherwise read the
    # bare 1.2 and call the amplitude 1.2 V.
    scaled = None
    if vcc is not None:
        scaled = re.search(
            r"(?<![A-Za-z0-9])V_?cm\s*=\s*"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:[×*·]|\\times)\s*V_?CC",
            text, re.IGNORECASE,
        )
    if scaled is not None:
        vcm = float(scaled.group(1)) * vcc
    else:
        vcm = _assignment(text, r"V_?cm|Vcm")

    if vcm is None and rf_context:
        # ``2 I_c1m R_p`` / ``2I_{c1m}R_p``: implicit or explicit multiplication.
        doubled_voltage = _assignment(text, _DOUBLED_PRODUCT)
        if doubled_voltage is not None:
            vcm = doubled_voltage / 2

    if vcm is None and vcc is not None and rp is not None:
        if ic1m is None and icmax is not None and alpha1 is not None:
            ic1m = alpha1 * icmax
        if ic1m is not None:
            vcm = ic1m * rp
    # 归一化 is a normalization instruction, not a different circuit; it must
    # not hide a real V_cm > V_CC violation.
    unrelated = re.search(r"差分|差动|双电源|共模|differential|common.mode|阻尼", text, re.IGNORECASE)
    if vcm is not None and vcc is not None and vcc > 0 and not unrelated and vcm > vcc + 1e-6:
        return (
            f"V_cm={vcm:g} V exceeds V_CC={vcc:g} V "
            "(resonant load voltage swing is physically impossible)"
        )
    voltage_utilization = re.search(r"电压利用|voltage.utilization|谐振功|谐振功率|resonant.*amplif", text, re.IGNORECASE)
    if xi is not None and xi > 1.0 + 1e-6 and voltage_utilization and not unrelated:
        return f"xi={xi:g} exceeds 1 (voltage utilization factor)"
    return None


def _gate_parameter_feasibility(spec: ReviewSpec) -> GateResult:
    details = [
        f"item {item.id}: {reason}"
        for item in spec.items
        if (reason := parameter_infeasibility(item))
    ]
    return GateResult(
        gate="G9-parameter-feasibility",
        passed=not details,
        message="parameters are physically feasible" if not details else "impossible parameters",
        details=details[:20],
    )


# SI prefixes accepted for the units below; everything is normalized to the
# base unit (V, A, W, Ω, Hz, F, s) before the comparison.
_SI_SCALE = {
    "": Decimal(1),
    "p": Decimal("1e-12"),
    "n": Decimal("1e-9"),
    "u": Decimal("1e-6"),
    "μ": Decimal("1e-6"),
    "µ": Decimal("1e-6"),
    "m": Decimal("1e-3"),
    "k": Decimal("1e3"),
    "M": Decimal("1e6"),
    "G": Decimal("1e9"),
}
_MAX_DECIMAL_POWER = 400
_QUANTITY_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.])"
    r"(?P<sign>[-+]?)"
    r"(?P<mantissa>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
    r"(?:[eE](?P<exponent>[-+]?\d+))?"
    r"(?:\s*(?:[×x*·]|\\times|\\cdot)\s*10\s*(?:\^|\*\*)\s*\{?\s*(?P<power>[-+]?\d+)\s*\}?)?"
    r"\s*(?:(?P<prefix>[pnumkMGμµ]?)(?P<unit>Hz|ohm|Ω|V|A|W|F|s))?"
    r"\s*(?P<percent>%|％)?"
    r"(?![\d.,]|[eE][-+]?\d)"
)


def _quantity_values(match: re.Match[str]) -> tuple[Decimal, ...]:
    """Exact values one numeric candidate may stand for, or ``()`` if malformed.

    Everything stays in ``Decimal``: routing the candidate through ``float``
    would make ``0.12345678901234568`` satisfy ``0.1234567890123456789``.

    ``Answer.value`` carries no unit, so ``1.592 MHz`` legitimately satisfies
    both ``1.592`` (the answer is stated in MHz, as this fixture does) and
    ``1.592e6`` (the answer is stated in Hz). Both readings are returned and
    compared exactly; only prefix arithmetic differs.
    """
    try:
        value = Decimal(match.group("mantissa").replace(",", ""))
        sign = match.group("sign") or ""
        if sign == "-":
            value = -value
        exponent = match.group("exponent")
        if exponent:
            if len(exponent.lstrip("+-")) > 6:
                return ()
            value = value.scaleb(int(exponent))
        power = match.group("power")
        if power:
            power_value = int(power)
            if abs(power_value) > _MAX_DECIMAL_POWER:
                return ()
            value = value * (Decimal(10) ** power_value)
        values = [value]
        if match.group("unit"):
            scaled = value * _SI_SCALE[match.group("prefix") or ""]
            if scaled != value:
                values.append(scaled)
        if match.group("percent"):
            values = [candidate / Decimal(100) for candidate in values]
        return tuple(values)
    except (ArithmeticError, InvalidOperation, ValueError):
        return ()


def _mentions_number(text: str, value: float | str) -> bool:
    """Whether ``text`` mentions ``value`` as a standalone number.

    A plain substring check lets a numeric answer of ``2`` match the ``12``
    inside an unrelated figure. A candidate only counts when neither
    neighbouring character is a digit, so ``12`` no longer satisfies ``2``.
    Accepted spellings: thousands separators (``1,200``), ``×10^n`` /
    ``\\times10^{n}``, percent (``50%``) and SI prefixes on V/A/W/Ω/Hz/F/s
    (``500 mV``, ``2 kV``, ``1 μF``, ``3 MHz``).
    """
    expected = Decimal(str(value))
    normalized = unicodedata.normalize("NFKC", text)
    for match in _QUANTITY_PATTERN.finditer(normalized):
        if any(candidate == expected for candidate in _quantity_values(match)):
            return True
    return False


def _gate_answers(spec: ReviewSpec) -> GateResult:
    details: list[str] = []
    by_id = {item.id: item for item in spec.items}
    for item in spec.items:
        if not item.explanation.strip():
            details.append(f"item {item.id}: explanation is empty")
            continue
        if item.kind == "single_choice":
            selected = item.options[item.answer.choice_index or 0].text
            if selected not in item.explanation:
                details.append(f"item {item.id}: explanation does not contain the correct option")
        elif item.kind == "numeric":
            value = item.answer.value
            if value is None:
                details.append(f"item {item.id}: numeric answer is missing")
            elif not _mentions_number(item.explanation, value):
                details.append(f"item {item.id}: explanation does not contain {value:g}")
        elif item.kind == "short_answer":
            answer_text = item.answer.text.strip()
            if (
                answer_text
                and answer_coverage(item.explanation, answer_text)
                <= ANSWER_COVERAGE_MIN
            ):
                details.append(
                    f"item {item.id}: explanation does not contain the short answer"
                )
        if item.variant_of is not None and item.variant_of not in by_id:
            details.append(f"item {item.id}: variant_of references a missing item")
    return GateResult(
        gate="G4-answer-key",
        passed=not details,
        message="answer keys agree with explanations" if not details else "answer key conflict",
        details=details[:20],
    )


def _gate_safety(spec: ReviewSpec) -> GateResult:
    details: list[str] = []
    # Include the quality report: warnings are client-supplied text that is
    # serialized into the export payload, so they must pass the same gate.
    for path, text in _strings(spec.model_dump(mode="json")):
        for pattern in _UNSAFE_PATTERNS:
            if pattern.search(text):
                details.append(f"{path}: unsafe pattern {pattern.pattern!r}")
    unique = list(dict.fromkeys(details))
    return GateResult(
        gate="G6-injection",
        passed=not unique,
        message="no executable strings" if not unique else "executable content rejected",
        details=unique[:20],
    )


def _gate_offline(spec: ReviewSpec) -> GateResult:
    details: list[str] = []
    for path, text in _strings(spec.model_dump(mode="json")):
        if _EXTERNAL_URL.search(text):
            details.append(f"{path}: external URL would break offline export")
    unique = list(dict.fromkeys(details))
    return GateResult(
        gate="G8-offline",
        passed=not unique,
        message="no external references" if not unique else "external reference found",
        details=unique[:20],
    )


_GATES = (
    _gate_schema,
    _gate_sources,
    _gate_math,
    _gate_math_balance,
    _gate_answers,
    _gate_parameter_feasibility,
    _gate_safety,
    _gate_offline,
)


def run_gates(spec: ReviewSpec) -> QualityReport:
    gates = [gate(spec) for gate in _GATES]
    warnings = list(spec.quality.warnings)
    multi_concept = [
        f"item {item.id}: multiple concepts; Phase 0.5 schedules "
        f"{item.concept_ids[0]} only"
        for item in spec.items
        if len(item.concept_ids) > 1
    ]
    warnings.extend(multi_concept[:20])
    for gate in gates:
        if gate.passed:
            continue
        warnings.append(f"{gate.gate}: {gate.message}")
    return QualityReport(
        passed=all(gate.passed for gate in gates),
        warnings=warnings,
        gates=gates,
    )


def apply_quality(spec: ReviewSpec) -> ReviewSpec:
    """Attach the latest gate report without changing the content hash."""
    report = run_gates(spec)
    return spec.model_copy(update={"quality": report})
