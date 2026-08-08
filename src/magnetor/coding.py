"""Branch C — the Lakatosian coding instrument (Part III of the critique document).

**What this is, and what it deliberately is not.** Part IV of the source document
states the position plainly: *"The corpus is unbuilt… the largest single piece of
unfunded work"* and *"No coding manual has been piloted."* So this builds the
**instrument** — a schema, a store, and the reliability statistic that decides
whether the schema works — not the study. Most Part III fields need full text and
human judgement (``use_novel``, ``excess_content``, ``power_regime``,
``analytic_dof``); nothing here fabricates them.

**Why an instrument is worth building before the corpus.** Section II.1 makes the
argument: *"A grammar becomes empirical the moment it is used to code a corpus."*
If independent coders cannot agree on ``level_sense`` or ``interp_schema_type``,
that is a real, publishable negative result about the framework — and III.6 is
explicit that a field below the reliability floor is *"a finding about the
framework, not a coding failure to be patched."* Krippendorff's alpha is therefore
the point of this module, not an accessory to it.

**Prefilled fields never assert more than was checked.** ``integrity_status``
prefills to ``retracted`` when OpenAlex flags it and otherwise to ``not-checked``,
never to ``clean``: Retraction Watch and PubPeer are not consulted here, and
recording an unchecked paper as clean would manufacture the very false precision
the architecture exists to avoid (C3 — absence is displayed, not imputed).

**Storage is append-only JSONL, one file per graph.** Coding is evidence about
coders as much as about papers, so an overwrite would destroy the disagreement
that alpha is computed from.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from magnetor.config import global_store_path
from magnetor.graph import query_hash

CODING_DIRNAME = "coding"

#: Environment override for who is coding. III.6 requires two or more independent
#: coders, so an unattributed record cannot contribute to reliability.
CODER_ENV = "MAGNETOR_CODER"

UNSET = "unset"

#: The Part III vocabularies. ``unset`` is always permitted and always the
#: default: a coder who has not judged a field must be distinguishable from one
#: who judged it and found nothing, which is the same reason the manual carries
#: ``none-of-the-above`` and ``unspecified`` as real values rather than blanks.
FIELDS: dict[str, tuple[str, ...]] = {
    "explanation_type": ("descriptive", "mechanistic", "normative", "none-of-the-above"),
    "level_sense": ("L_scale", "L_comp", "L_const", "L_ideal", "L_dyn", "unspecified"),
    "interp_schema_type": ("T0", "T1", "T2", "T3"),
    "interp_invertible": ("yes", "no", "not-stated"),
    "data_or_phenomenon": ("datum-claim", "phenomenon-claim"),
    "normative_subscript": ("N_sel", "N_asc", "N_fit", "not-applicable"),
    "normative_presented_as": ("N_sel", "N_asc", "N_fit", "not-applicable"),
    "claim_type": ("stipulative", "formal", "empirical", "imperative"),
    "core_or_belt": ("core", "belt", "undeclared"),
    "imperative_form": ("categorical", "conditional", "not-applicable"),
    "integrity_status": (
        "clean", "corrected", "expression-of-concern", "retracted",
        "contested-unresolved", "not-checked",
    ),
    "confirmation_independence": ("same-lab", "same-lineage", "independent", "adversarial"),
    "replication_attempt_status": ("none-attempted", "succeeded", "failed", "mixed"),
    # --- III.3, per-revision: the fields the II.9 gate runs on ---
    "independently_measurable": ("yes", "no"),
    "excess_content": ("yes", "no"),
    "use_novel": ("yes", "no", "undeterminable"),
    "parsimony_out_of_sample": ("improved", "neutral", "worse", "not-applicable"),
    "form_derived_independently": ("yes", "no"),
    "novelty_sense": ("temporal", "use-novel", "both", "neither"),
}

#: The II.9 gate's three necessary criteria, in order. Criteria 4 and 5
#: (parsimony, non-arbitrary form) are strength grades and never decide a verdict.
_GATE = ("independently_measurable", "excess_content", "use_novel")

PROGRESSIVE = "progressive"
STAGNANT = "stagnant"
DEGENERATING = "degenerating"
UNDETERMINED = "undetermined"

#: How far a verdict can be trusted, which is a separate question from what the
#: verdict says. The gate is deterministic; its inputs are not, and a crisp label
#: over unmeasured coding manufactures confidence the evidence has not earned.
UNVALIDATED = "unvalidated"  # agreement never measured
UNRELIABLE = "unreliable"  # measured, below the floor
USABLE = "usable"
FIRM = "firm"

#: Years without a gate-passing revision before a programme reads as stalled.
#: II.2 requires a window to be *stipulated and declared a convention*, because
#: Lakatos gives no rule for when a lull becomes degeneration — the defect he
#: charged Kuhn with. A decade is the source's own example, not a calibration.
DEFAULT_DEGENERATION_WINDOW = 10

#: The unit mismatch, carried with every output rather than buried in a docstring.
#: III.1 codes the *claim* as a ``(model, context, problem)`` triple and III.3's
#: fields are per-*revision*; this instrument attaches them to a paper. A paper is
#: not a revision: one can contain several, and one revision can span papers.
CODING_UNIT_CAVEAT = (
    "Coded per paper. The source's unit is the claim, and the gate's fields are "
    "per-revision, so a paper carrying more than one revision is coded as one."
)

#: Reliability floors from III.6, carried as declared conventions.
ALPHA_FLOOR = 0.67
ALPHA_FIRM = 0.80


#: What a field does to the verdict. The distinction is the whole navigational
#: problem: thirteen fields look equally weighty on a form, but only three decide
#: anything, four can void a decision, and the rest are recorded for the corpus.
GATE = "gate"
DEFEATER = "defeater"
CONTEXT = "context"


@dataclass(frozen=True, slots=True)
class FieldGuide:
    """Everything a coder needs to answer one field without guessing."""

    question: str  # what is actually being asked
    role: str  # GATE | DEFEATER | CONTEXT
    look_for: str  # where in the paper the answer lives
    effect: str  # what this does to the outcome
    values: dict[str, str]  # value -> what choosing it asserts


FIELD_GUIDE: dict[str, FieldGuide] = {
    # --- the gate: these three, and only these three, decide the verdict ---
    "independently_measurable": FieldGuide(
        question="Can what this revision added be measured by anything other than "
        "the fit that motivated it?",
        role=GATE,
        look_for="An instrument, dataset or observation that could confirm the added "
        "term without reusing the anomaly it was introduced to explain. The source's "
        "worked case: a 'regret' term was added to a foraging model, and independent "
        "neural correlates were later found — so it passes.",
        effect="No here means degenerating, whatever else is true.",
        values={
            "yes": "An independent operation exists that could confirm the addition.",
            "no": "The addition is visible only in the fit that produced it — the "
            "textbook ad hoc rescue.",
        },
    ),
    "excess_content": FieldGuide(
        question="Does the revised theory predict anything outside the anomaly it "
        "was built to explain?",
        role=GATE,
        look_for="A claim following from the revision about some other condition, "
        "population, scale or domain. Explaining the original anomaly better does "
        "not count — that is what it was built for.",
        effect="No here means stagnant: checkable, but content-free.",
        values={
            "yes": "The revision entails at least one prediction beyond its anomaly.",
            "no": "It accounts for its anomaly and nothing further.",
        },
    ),
    "use_novel": FieldGuide(
        question="Was that prediction about facts not used in constructing the "
        "revision?",
        role=GATE,
        look_for="Whether the confirming evidence was already known and fitted to. "
        "Publication dates help but do not settle it: a fact can predate the revision "
        "and still not have been used in building it.",
        effect="No means stagnant. Undeterminable means undetermined — the source is "
        "explicit that this judgement requires reading, not metadata.",
        values={
            "yes": "The confirming facts were not used in building the revision.",
            "no": "The revision was fitted to the facts that now support it.",
            "undeterminable": "The record does not say which came first.",
        },
    ),
    # --- defeaters: they cannot create a pass, only void one ---
    "core_or_belt": FieldGuide(
        question="Is this claim part of the programme's unrevisable core, or its "
        "testable protective belt?",
        role=DEFEATER,
        look_for="Whether abandoning the claim would end the programme (core) or "
        "merely revise it (belt).",
        effect="Decides whether an integrity problem defeats the verdict: only "
        "core-supporting evidence does.",
        values={
            "core": "Abandoning this would end the programme.",
            "belt": "A testable auxiliary that can be revised.",
            "undeclared": "The paper does not mark which it is.",
        },
    ),
    "integrity_status": FieldGuide(
        question="What is the publication's integrity record?",
        role=DEFEATER,
        look_for="Retraction notices, expressions of concern, corrections, PubPeer "
        "threads. Prefilled as retracted only when OpenAlex flags it.",
        effect="Defeats a passing verdict when the paper is also core-supporting.",
        values={
            "clean": "Checked, and no notice found.",
            "corrected": "A correction has been issued.",
            "expression-of-concern": "The publisher has raised a concern.",
            "retracted": "Withdrawn.",
            "contested-unresolved": "Disputed, with nothing settled.",
            "not-checked": "Nobody has looked. Never assume this means clean.",
        },
    ),
    "confirmation_independence": FieldGuide(
        question="Who produced the confirming evidence?",
        role=DEFEATER,
        look_for="Author overlap and mentorship lineage between the original claim "
        "and its confirmations.",
        effect="Same-lab confirmation defeats the verdict: corroboration by the "
        "originating group is weak evidence, which Lakatos's scheme cannot register.",
        values={
            "same-lab": "Confirmed only by the group that made the claim.",
            "same-lineage": "Confirmed within one mentorship or collaboration tree.",
            "independent": "Confirmed by an unrelated group.",
            "adversarial": "Confirmed under a design agreed with sceptics — the "
            "strongest and rarest.",
        },
    ),
    "replication_attempt_status": FieldGuide(
        question="Has anyone attempted to replicate it?",
        role=DEFEATER,
        look_for="Replication attempts, whatever their outcome. 'No failed "
        "replications' is uninformative if nobody tried.",
        effect="None-attempted defeats the verdict: absence of falsification is not "
        "corroboration.",
        values={
            "none-attempted": "Untested. Nothing has been risked.",
            "succeeded": "Replicated.",
            "failed": "A replication attempt failed.",
            "mixed": "Attempts disagree.",
        },
    ),
    # --- context: recorded for the corpus, no effect on this verdict ---
    "explanation_type": FieldGuide(
        question="What kind of explanation is being offered?",
        role=CONTEXT,
        look_for="Whether the claim describes a pattern, gives a mechanism, or says "
        "what something is for.",
        effect="Recorded. This is the field whose coder agreement tests the "
        "framework itself.",
        values={
            "descriptive": "Characterises a phenomenon without explaining it.",
            "mechanistic": "Says how the system produces it.",
            "normative": "Says what it is for, or what would be optimal.",
            "none-of-the-above": "Fits none — a real and reportable outcome.",
        },
    ),
    "level_sense": FieldGuide(
        question="In which sense is 'level' being used?",
        role=CONTEXT,
        look_for="Whether the paper means physical scale, substrate abstraction, "
        "part-whole constitution, degree of idealisation, or dynamical autonomy.",
        effect="Recorded. Drift in the unspecified rate is itself a signal of a "
        "field clarifying.",
        values={
            "L_scale": "Spatiotemporal grain.",
            "L_comp": "Substrate abstraction — survives an implementation swap.",
            "L_const": "Constitutive part-whole. Contested.",
            "L_ideal": "Degree of idealisation.",
            "L_dyn": "Dynamically autonomous coarse-graining.",
            "unspecified": "The paper does not say. A finding, not a blank.",
        },
    ),
    "interp_schema_type": FieldGuide(
        question="How does the model connect to the world?",
        role=CONTEXT,
        look_for="The stated mapping between model terms and measured quantities.",
        effect="Recorded. Mature theories convert T3 into T1 over time.",
        values={
            "T0": "Instrument chain — model term to measured signal.",
            "T1": "Denotation — term to a quantity with units.",
            "T2": "Restriction — what is deliberately not tracked.",
            "T3": "Analogy — structural correspondence without denotation.",
        },
    ),
    "data_or_phenomenon": FieldGuide(
        question="Is the claim about the data, or about the phenomenon inferred "
        "from it?",
        role=CONTEXT,
        look_for="Whether the claim would survive a different instrument.",
        effect="Recorded. Datum-claims are predicted to replicate worse.",
        values={
            "datum-claim": "About this instrument's output.",
            "phenomenon-claim": "About a stable feature inferred from data.",
        },
    ),
    "normative_subscript": FieldGuide(
        question="Which sense of 'goal' is meant?",
        role=CONTEXT,
        look_for="Whether the goal was selected for, ascribed by the modeller, or "
        "estimated from behaviour.",
        effect="Recorded. Only N_sel answers why the phenomenon exists.",
        values={
            "N_sel": "The goal the system was selected for.",
            "N_asc": "A goal ascribed as a predictive heuristic.",
            "N_fit": "An objective function estimated from behaviour.",
            "not-applicable": "No normative claim.",
        },
    ),
    "claim_type": FieldGuide(
        question="What kind of claim is it?",
        role=CONTEXT,
        look_for="Whether it stipulates a meaning, derives a result, reports an "
        "observation, or tells someone what to do.",
        effect="Recorded. Separates what could be tested from what was defined.",
        values={
            "stipulative": "True by definition.",
            "formal": "Derived mathematically.",
            "empirical": "Answerable by observation.",
            "imperative": "A recommendation.",
        },
    ),
}


def fields_by_role(role: str) -> tuple[str, ...]:
    """Field names carrying a given role, in the guide's order."""
    return tuple(name for name, guide in FIELD_GUIDE.items() if guide.role == role)


class CodingError(ValueError):
    """A value outside its field's vocabulary."""


@dataclass(frozen=True, slots=True)
class CodedClaim:
    """One coder's judgement about one paper, in one graph."""

    graph_query: str
    paper_id: str
    coder: str
    coded_at: str
    values: dict[str, str] = field(default_factory=dict)
    note: str = ""

    def value(self, name: str) -> str:
        return self.values.get(name, UNSET)


def current_coder() -> str:
    return os.environ.get(CODER_ENV, "").strip() or "anonymous"


def validate(values: Mapping[str, str]) -> dict[str, str]:
    """Reject anything outside the vocabularies; unknown fields are an error too."""
    clean: dict[str, str] = {}
    for name, value in values.items():
        if name not in FIELDS:
            raise CodingError(f"unknown coding field: {name!r}")
        if value != UNSET and value not in FIELDS[name]:
            allowed = ", ".join((*FIELDS[name], UNSET))
            raise CodingError(f"{name}={value!r} is not one of: {allowed}")
        clean[name] = value
    return clean


def prefill(node: Mapping[str, object]) -> dict[str, str]:
    """Defaults derivable from what the graph already holds — and nothing more.

    Only ``integrity_status`` is derivable, and only in one direction: OpenAlex's
    ``is_retracted`` establishes a retraction but its absence establishes nothing,
    so the other branch is ``not-checked`` rather than ``clean``.
    """
    return {
        "integrity_status": "retracted" if node.get("is_retracted") else "not-checked",
        **({"claim_type": "empirical"} if node.get("is_review") is False else {}),
    }


def _root(coding_dir: Path | None) -> Path:
    return coding_dir or global_store_path(CODING_DIRNAME)


def _path(graph_query: str, coding_dir: Path | None) -> Path:
    return _root(coding_dir) / f"{query_hash(graph_query)}.jsonl"


def record(
    graph_query: str,
    paper_id: str,
    values: Mapping[str, str],
    *,
    coder: str | None = None,
    note: str = "",
    coding_dir: Path | None = None,
    now: dt.datetime | None = None,
) -> CodedClaim:
    """Append one coding decision. Never overwrites: disagreement is the data."""
    claim = CodedClaim(
        graph_query=graph_query,
        paper_id=paper_id,
        coder=coder or current_coder(),
        coded_at=(now or dt.datetime.now(dt.UTC)).isoformat(),
        values=validate(values),
        note=note,
    )
    path = _path(graph_query, coding_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(claim)) + "\n")
    return claim


def load(graph_query: str, *, coding_dir: Path | None = None) -> tuple[CodedClaim, ...]:
    """Every recorded judgement for a graph, oldest first. Corrupt lines are skipped."""
    path = _path(graph_query, coding_dir)
    if not path.exists():
        return ()
    claims: list[CodedClaim] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("paper_id"):
            claims.append(
                CodedClaim(
                    graph_query=str(data.get("graph_query", graph_query)),
                    paper_id=str(data["paper_id"]),
                    coder=str(data.get("coder", "anonymous")),
                    coded_at=str(data.get("coded_at", "")),
                    values={
                        k: str(v) for k, v in (data.get("values") or {}).items()
                    },
                    note=str(data.get("note", "")),
                )
            )
    return tuple(claims)


def latest_by_coder(claims: Sequence[CodedClaim], field_name: str) -> dict[str, dict[str, str]]:
    """``{paper_id: {coder: value}}`` using each coder's most recent judgement.

    A coder who revises keeps one vote; the superseded record stays on disk for
    audit but must not count twice toward agreement.
    """
    latest: dict[tuple[str, str], CodedClaim] = {}
    for claim in claims:
        if claim.value(field_name) == UNSET:
            continue
        key = (claim.paper_id, claim.coder)
        current = latest.get(key)
        if current is None or claim.coded_at >= current.coded_at:
            latest[key] = claim
    out: dict[str, dict[str, str]] = defaultdict(dict)
    for (paper_id, coder), claim in latest.items():
        out[paper_id][coder] = claim.value(field_name)
    return dict(out)


def krippendorff_alpha(ratings: Mapping[str, Mapping[str, str]]) -> float | None:
    """Nominal Krippendorff's alpha, or ``None`` when it is undefined.

    Undefined rather than zero when fewer than two units carry two or more
    ratings, or when every rating is identical: with no variance there is no
    expected disagreement to divide by, and returning a number there would invent
    a reliability that was never measured.
    """
    units = [list(v.values()) for v in ratings.values() if len(v) >= 2]
    if len(units) < 2:
        return None

    coincidences: Counter[tuple[str, str]] = Counter()
    for values in units:
        pairs = len(values) - 1
        for i, first in enumerate(values):
            for j, second in enumerate(values):
                if i != j:
                    coincidences[(first, second)] += 1.0 / pairs  # type: ignore[assignment]

    totals: Counter[str] = Counter()
    for (first, _second), weight in coincidences.items():
        totals[first] += weight
    n = sum(totals.values())
    if n <= 1:
        return None

    observed = sum(w for (a, b), w in coincidences.items() if a != b)
    expected = sum(
        totals[a] * totals[b] for a in totals for b in totals if a != b
    ) / (n - 1)
    if expected <= 0:
        return None
    return 1.0 - (observed / expected)


@dataclass(frozen=True, slots=True)
class FieldReliability:
    field_name: str
    alpha: float | None
    units: int  # papers with two or more independent judgements
    coders: int

    @property
    def verdict(self) -> str:
        if self.alpha is None:
            return "not enough double-coded papers"
        if self.alpha >= ALPHA_FIRM:
            return "firm"
        if self.alpha >= ALPHA_FLOOR:
            return "usable"
        return "below floor — a finding about the framework, not a coding error"


def reliability(
    claims: Sequence[CodedClaim], field_name: str
) -> FieldReliability:
    """Inter-coder agreement for one field (III.6)."""
    ratings = latest_by_coder(claims, field_name)
    coders = {coder for votes in ratings.values() for coder in votes}
    return FieldReliability(
        field_name=field_name,
        alpha=krippendorff_alpha(ratings),
        units=sum(1 for votes in ratings.values() if len(votes) >= 2),
        coders=len(coders),
    )


def reliability_report(claims: Sequence[CodedClaim]) -> tuple[FieldReliability, ...]:
    """Agreement across every field that anyone has actually coded."""
    return tuple(
        reliability(claims, name)
        for name in FIELDS
        if any(claim.value(name) != UNSET for claim in claims)
    )


def coverage(claims: Sequence[CodedClaim], field_name: str) -> int:
    """Distinct papers with at least one judgement on this field."""
    return len({c.paper_id for c in claims if c.value(field_name) != UNSET})


@dataclass(frozen=True, slots=True)
class Verdict:
    """A II.9 judgement on one revision, with the reasoning kept attached."""

    verdict: str  # PROGRESSIVE | STAGNANT | DEGENERATING | UNDETERMINED
    reason: str
    defeated_by: str = ""

    @property
    def is_defeated(self) -> bool:
        return bool(self.defeated_by)


def judge(values: Mapping[str, str]) -> Verdict:
    """Apply the II.9 gate: progressive only if criteria 1-3 all pass.

    The gate is deliberately not a score. Criterion 4 (parsimony) and criterion 5
    (non-arbitrary functional form) are strength grades in the source and stay
    that way here, because averaging a necessary condition against a grade is how
    a failed criterion gets bought off by a strong one.

    Three outcomes rather than two. *Stagnant* — measurable but content-free —
    is the category the binary forces false verdicts on, and the source calls it
    "the largest and most under-recognised". *Undetermined* is returned whenever
    the gate cannot be evaluated, because III.4E is explicit that absence of
    falsification is not corroboration and must never be read as progress.
    """
    measurable = values.get("independently_measurable", UNSET)
    content = values.get("excess_content", UNSET)
    novel = values.get("use_novel", UNSET)

    if any(values.get(name, UNSET) == UNSET for name in _GATE):
        missing = [name for name in _GATE if values.get(name, UNSET) == UNSET]
        return Verdict(
            UNDETERMINED,
            f"not yet coded: {', '.join(missing)}. Absence of falsification is not "
            "corroboration, so this is undetermined rather than progressive.",
        )

    defeat = _defeater(values)
    if measurable == "no":
        return Verdict(
            DEGENERATING,
            "the added term is not measurable by any operation other than the fit "
            "that motivated it (criterion 1).",
            defeat,
        )
    if content == "no" or novel == "no":
        failed = "excess content" if content == "no" else "use-novelty"
        return Verdict(
            STAGNANT,
            f"measurable, but fails {failed} (criteria 2-3): the revision is "
            "checkable and yet predicts nothing outside the anomaly it was built for.",
            defeat,
        )
    if novel == "undeterminable":
        return Verdict(
            UNDETERMINED,
            "use-novelty could not be determined from the available record, and "
            "the source is explicit that this requires reading rather than metadata.",
            defeat,
        )
    return Verdict(
        PROGRESSIVE,
        "independently measurable, carries excess content, and that content is "
        "use-novel (criteria 1-3).",
        defeat,
    )


def _defeater(values: Mapping[str, str]) -> str:
    """III.4E defeaters — binary and auditable, never a weighted downgrade.

    A weighted downgrade would reintroduce exactly the invented precision II.4
    rejects, so a verdict is either defeated or it is not.
    """
    if values.get("integrity_status") in ("retracted", "contested-unresolved") and values.get(
        "core_or_belt"
    ) == "core":
        return "rests on core-supporting evidence that is retracted or contested"
    if values.get("confirmation_independence") == "same-lab":
        return "every confirmation comes from the originating lab"
    if values.get("replication_attempt_status") == "none-attempted":
        return "no replication has been attempted, so nothing has been risked"
    return ""


def programme_verdict(
    claims: Sequence[CodedClaim],
    *,
    years: Mapping[str, int | None] | None = None,
    window_years: int = DEFAULT_DEGENERATION_WINDOW,
    as_of: int | None = None,
) -> Verdict:
    """Roll per-paper judgements into one programme-level reading, then run the clock.

    Two stipulations, both declared rather than derived:

    **The roll-up is pessimistic.** One degenerating revision outweighs several
    progressive ones, because Lakatos's question is whether the programme is still
    generating novel content, not how much of its output is respectable. Neither
    Lakatos nor the source states this rule — it is a convention of this
    implementation and is named as one so it can be disagreed with specifically.

    **The clock needs publication years.** Lakatos's contribution over Kuhn is a
    *temporal* index, so a gate without one is the framework with its distinctive
    part removed. Pass ``years`` and a programme whose last gate-passing revision
    predates ``as_of - window_years`` reads stagnant however good those revisions
    were. Without years the clock is skipped and the reason says so, because
    silently omitting a criterion is how a partial instrument comes to look whole.

    ``as_of`` defaults to the current year. For a historical study it must be set
    to the cut-off date being coded at (II.14), or every past programme reads as
    stalled by construction.
    """
    judged = [(claim, judge(claim.values)) for claim in claims]
    decided = [(c, v) for c, v in judged if v.verdict != UNDETERMINED]
    if not decided:
        return Verdict(UNDETERMINED, "no revision has been coded through the gate.")
    counts = Counter(v.verdict for _c, v in decided)
    defeated = [v for _c, v in decided if v.is_defeated]
    summary = ", ".join(f"{n} {verdict}" for verdict, n in counts.most_common())

    if counts[PROGRESSIVE] == 0:
        worst = DEGENERATING if counts[DEGENERATING] else STAGNANT
        return Verdict(worst, f"no coded revision passed the gate ({summary}).")
    if counts[DEGENERATING]:
        return Verdict(
            STAGNANT,
            f"progress is mixed with degeneration ({summary}); a programme is not "
            "progressive because some of its revisions are. [Stipulated roll-up: "
            "one degenerating revision holds the programme back.]",
        )

    tail = (
        f" {len(defeated)} verdict(s) defeated on evidential grounds." if defeated else ""
    )
    clock = _clock(decided, years=years, window_years=window_years, as_of=as_of)
    if clock is not None:
        return clock
    note = (
        ""
        if years is not None
        else " Clock not run: no publication years supplied, so this reports the "
        "gate only and cannot say whether the programme is still moving."
    )
    return Verdict(PROGRESSIVE, f"{summary}.{tail}{note}")


def _clock(
    decided: Sequence[tuple[CodedClaim, Verdict]],
    *,
    years: Mapping[str, int | None] | None,
    window_years: int,
    as_of: int | None,
) -> Verdict | None:
    """Stagnation by lull (II.2), or ``None`` when the clock cannot or need not fire."""
    if years is None:
        return None
    passing = [
        years.get(claim.paper_id)
        for claim, verdict in decided
        if verdict.verdict == PROGRESSIVE
    ]
    known = [year for year in passing if isinstance(year, int)]
    if not known:
        return Verdict(
            UNDETERMINED,
            "no publication year is known for any gate-passing revision, so the "
            "degeneration window cannot be applied.",
        )
    latest = max(known)
    reference = as_of if as_of is not None else dt.datetime.now(dt.UTC).year
    if reference - latest > window_years:
        return Verdict(
            STAGNANT,
            f"the last gate-passing revision dates to {latest}, more than the "
            f"stipulated {window_years}-year window before {reference}. Passing the "
            "gate in the past is not the same as still moving. [Window is a declared "
            "convention, not a calibration.]",
        )
    return None


@dataclass(frozen=True, slots=True)
class QualifiedVerdict:
    """A verdict together with how far its inputs can carry it."""

    verdict: Verdict
    standing: str  # UNVALIDATED | UNRELIABLE | USABLE | FIRM
    gate_reliability: tuple[FieldReliability, ...]
    caveat: str

    @property
    def is_trustworthy(self) -> bool:
        return self.standing in (USABLE, FIRM)


def qualify(verdict: Verdict, claims: Sequence[CodedClaim]) -> QualifiedVerdict:
    """Attach the measured agreement on the fields the verdict was computed from.

    The gate is a deterministic function of judgements that are not themselves
    deterministic, so the label it emits is never stronger than the coding beneath
    it — and being crisp, it reads stronger. Standing is the *worst* of the three
    gate fields rather than an average: a verdict is only as good as its weakest
    necessary input, and averaging would let one well-agreed field mask another
    nobody can reproduce.
    """
    measured = tuple(reliability(claims, name) for name in _GATE)
    alphas = [item.alpha for item in measured]

    if any(alpha is None for alpha in alphas):
        standing = UNVALIDATED
        caveat = (
            "Agreement on the gate's fields has never been measured — no paper here "
            "carries two independent codings. This verdict is one coder's reasoning "
            "made explicit, not a measurement. " + CODING_UNIT_CAVEAT
        )
    elif any(alpha < ALPHA_FLOOR for alpha in alphas if alpha is not None):
        weakest = min((a for a in alphas if a is not None), default=0.0)
        standing = UNRELIABLE
        caveat = (
            f"Coders do not reliably agree on at least one gate field (lowest alpha "
            f"{weakest:.2f}, floor {ALPHA_FLOOR}). III.6 treats that as a finding "
            "about the framework rather than a coding error, so read the "
            "disagreement before the verdict. " + CODING_UNIT_CAVEAT
        )
    elif all(alpha >= ALPHA_FIRM for alpha in alphas if alpha is not None):
        standing = FIRM
        caveat = CODING_UNIT_CAVEAT
    else:
        standing = USABLE
        caveat = (
            f"Agreement clears the {ALPHA_FLOOR} floor but not the {ALPHA_FIRM} bar "
            "for firm conclusions. " + CODING_UNIT_CAVEAT
        )
    return QualifiedVerdict(
        verdict=verdict, standing=standing, gate_reliability=measured, caveat=caveat
    )


def to_csv(claims: Iterable[CodedClaim]) -> str:
    """Flat table, one row per recorded judgement, every field a column."""
    import csv

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["paper_id", "coder", "coded_at", *FIELDS, "note"])
    for claim in claims:
        writer.writerow(
            [claim.paper_id, claim.coder, claim.coded_at]
            + [claim.value(name) for name in FIELDS]
            + [claim.note]
        )
    return buffer.getvalue()
