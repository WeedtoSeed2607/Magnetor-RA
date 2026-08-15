"""Can two readers agree on what a group of papers is about?

Stage 1 of the Synthesizer question, and it involves no model at all. Before
asking whether a machine can name a cluster well, there has to be something for
*well* to mean. If two competent readers, shown the same papers, cannot agree on
what the group is about, then the task has no ground truth and no model can be
scored against it — which is a real answer, obtained in an afternoon, rather than
a reason to install anything.

**Why the clusters are frozen.** Grouping is deterministic, but it is a function
of the corpus, and the corpus grows. Two readers coding a week apart would
otherwise name different groups and their disagreement would be an artefact of
the harvest. :func:`freeze` snapshots the membership once; every reader afterwards
sees exactly that.

**Why the computed words are withheld while naming.** A reader shown
*"decoherence, superconducting, qubits"* will paraphrase it, and the experiment
would measure agreement with the word labeller rather than agreement about the
papers. Naming shows titles only. The computed label is stored in the snapshot
for later comparison, and :func:`to_name` does not return it.

**Why the statistic is computed over judging, not naming.** Krippendorff's alpha
needs a categorical variable and free-text names are not one. So there are two
passes: everyone writes a name, then everyone rules on every name —
``accepts`` / ``partial`` / ``rejects`` — with authorship stripped and the order
shuffled per judge. Alpha is measured over those rulings, which are nominal, and
answers the question directly: do these readers agree about what counts as a
correct description of this group?

That second pass is also exactly what stage 2 needs. A machine-written name is
added to the same pool, and because :class:`Candidate` carries no author, no
judge can tell which one it is.

**Storage is append-only JSONL**, for the reason ``coding.py`` gives: an
overwrite would destroy the disagreement the statistic is computed from.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from magnetor.coding import current_coder, krippendorff_alpha
from magnetor.config import global_store_path

NAMING_DIRNAME = "naming"

#: Rulings a judge can give a name. ``PARTIAL`` is a real answer, not a fence to
#: sit on: a name that catches half a group is a different thing from one that
#: misses it, and collapsing the two would hide the most informative case.
ACCEPTS = "accepts"
PARTIAL = "partial"
REJECTS = "rejects"
VERDICTS = (ACCEPTS, PARTIAL, REJECTS)

VERDICT_PROMPT = {
    ACCEPTS: "Yes — this names what the group is about.",
    PARTIAL: "Partly — it fits some of these papers but not the group.",
    REJECTS: "No — this is not what the group is about.",
}

#: Marks a name written by a person rather than a model. Stored, never shown to a
#: judge: a name's author must not be visible while it is being ruled on.
HUMAN = "human"


@dataclass(frozen=True, slots=True)
class FrozenCluster:
    """One group, fixed so that every reader names the same thing."""

    cluster_id: str
    paper_ids: tuple[str, ...]  # nearest the group's centre first
    #: The word label the clusterer produced. Kept for comparison afterwards and
    #: deliberately not shown while naming.
    computed_label: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NamingSet:
    """A frozen batch of clusters, the unit an experiment runs over."""

    set_id: str
    domain: str
    created_at: str
    clusters: tuple[FrozenCluster, ...]

    def cluster(self, cluster_id: str) -> FrozenCluster | None:
        return next((c for c in self.clusters if c.cluster_id == cluster_id), None)


@dataclass(frozen=True, slots=True)
class Name:
    """One author's one-line name for one cluster."""

    set_id: str
    cluster_id: str
    coder: str
    text: str
    written_at: str
    #: ``HUMAN``, or a model identifier once stage 2 runs.
    source: str = HUMAN

    @property
    def name_id(self) -> str:
        """Stable identity: one author writes at most one name per cluster."""
        return _digest(self.set_id, self.cluster_id, self.coder)


@dataclass(frozen=True, slots=True)
class Candidate:
    """A name put to a judge, with every trace of its author removed.

    The absent fields are the point. There is no ``coder`` and no ``source`` here,
    so a judge cannot rule differently on a colleague's name or a machine's.
    """

    name_id: str
    cluster_id: str
    text: str


@dataclass(frozen=True, slots=True)
class Judgement:
    """One judge's ruling on one name."""

    set_id: str
    cluster_id: str
    name_id: str
    judge: str
    verdict: str
    judged_at: str


@dataclass(frozen=True, slots=True)
class Agreement:
    """How far the judges agreed, and on how much."""

    alpha: float | None
    judges: int
    units: int  # names ruled on by two or more judges
    #: Rulings that were only ever made by one judge, so they cannot contribute.
    single_judged: int
    verdict_spread: Mapping[str, int] = field(default_factory=dict)

    @property
    def measurable(self) -> bool:
        return self.alpha is not None

    def summary(self) -> str:
        """One plain sentence a reader can act on."""
        if self.alpha is None:
            if self.judges < 2:
                return (
                    f"Only {self.judges} person has ruled on these names. Agreement "
                    "needs two people judging the same names independently — a "
                    "single reader's consistency is not the same measurement."
                )
            return (
                f"{self.units} name(s) have been ruled on by two or more people, "
                "which is too few to measure agreement."
            )
        return (
            f"Agreement between {self.judges} readers over {self.units} names: "
            f"alpha {self.alpha:.2f}."
        )


def visible(text: str) -> str:
    """``text`` with invisible formatting characters removed, then trimmed.

    ``str.strip()`` does not remove a byte-order mark — ``"\\ufeff".isspace()`` is
    False — so a name carrying one reads as non-empty and would be stored as a
    real answer. Found by piping into the CLI, which is how a coder pasting from a
    document would hit it. Dropping the whole ``Cf`` (format) category also covers
    zero-width joiners and directional marks.
    """
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf").strip()


def _digest(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:16]


def _root(naming_dir: Path | None) -> Path:
    return naming_dir or global_store_path(NAMING_DIRNAME)


def _set_path(set_id: str, naming_dir: Path | None) -> Path:
    return _root(naming_dir) / f"{set_id}.set.json"


def _log_path(set_id: str, naming_dir: Path | None) -> Path:
    return _root(naming_dir) / f"{set_id}.jsonl"


def freeze(
    domain: str,
    clusters: Sequence[tuple[Sequence[str], Sequence[str]]],
    *,
    naming_dir: Path | None = None,
    now: dt.datetime | None = None,
) -> NamingSet:
    """Snapshot clusters as ``(paper_ids, computed_label)`` pairs, once.

    The set id is derived from the membership, so freezing the same grouping
    twice gives the same set rather than a second copy to reconcile.
    """
    moment = (now or dt.datetime.now(dt.UTC)).isoformat()
    frozen: list[FrozenCluster] = []
    for paper_ids, label in clusters:
        ids = tuple(str(p) for p in paper_ids)
        if not ids:
            continue
        frozen.append(
            FrozenCluster(
                cluster_id=_digest(domain, *ids),
                paper_ids=ids,
                computed_label=tuple(str(w) for w in label),
            )
        )
    set_id = _digest(domain, *(c.cluster_id for c in frozen))
    naming_set = NamingSet(
        set_id=set_id, domain=domain, created_at=moment, clusters=tuple(frozen)
    )
    path = _set_path(set_id, naming_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "set_id": naming_set.set_id,
                "domain": naming_set.domain,
                "created_at": naming_set.created_at,
                "clusters": [
                    {
                        "cluster_id": c.cluster_id,
                        "paper_ids": list(c.paper_ids),
                        "computed_label": list(c.computed_label),
                    }
                    for c in naming_set.clusters
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return naming_set


def bundle(naming_set: NamingSet, titles: Mapping[str, str]) -> dict[str, object]:
    """Everything a reader needs to do this task, and nothing else.

    Written so the public naming app can run with no corpus, no vector index, no
    API key and no harvest — the four things that make the full dashboard unsafe
    to expose. Titles only: this carries no abstracts, so the bundle holds nothing
    a reader could not already see on the source's own listing page.

    ``computed_label`` is deliberately excluded. It is in the local snapshot for
    comparison afterwards, but shipping it to the app that shows readers the
    papers would put the words one click from the person meant not to see them.
    """
    return {
        "set_id": naming_set.set_id,
        "domain": naming_set.domain,
        "created_at": naming_set.created_at,
        "clusters": [
            {"cluster_id": c.cluster_id, "paper_ids": list(c.paper_ids)}
            for c in naming_set.clusters
        ],
        "titles": {
            pid: titles.get(pid, pid)
            for c in naming_set.clusters
            for pid in c.paper_ids
        },
    }


def read_bundle(path: Path) -> tuple[NamingSet, dict[str, str]]:
    """Load a bundle written by :func:`bundle`.

    The clusters come back with an empty ``computed_label`` because the bundle
    never carried one — an absence the type makes visible rather than a blank
    that looks like "no distinctive words were found".
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    clusters = tuple(
        FrozenCluster(
            cluster_id=str(c.get("cluster_id", "")),
            paper_ids=tuple(str(p) for p in c.get("paper_ids", [])),
            computed_label=(),
        )
        for c in data.get("clusters", [])
        if isinstance(c, dict)
    )
    naming_set = NamingSet(
        set_id=str(data.get("set_id", "")),
        domain=str(data.get("domain", "")),
        created_at=str(data.get("created_at", "")),
        clusters=clusters,
    )
    raw = data.get("titles")
    titles = {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}
    return naming_set, titles


def export_lines(set_id: str, coder: str, *, naming_dir: Path | None = None) -> str:
    """One reader's own records as JSONL, for carrying back from a hosted app.

    The log is append-only, so merging a remote reader's work is concatenation —
    there is no state to reconcile and no ordering to preserve. Filtered to one
    coder so a reader returns their own answers and cannot carry anyone else's.
    """
    wanted = []
    for record in _records(set_id, naming_dir):
        who = record.get("coder") or record.get("judge")
        if who == coder:
            wanted.append(json.dumps(record, sort_keys=True))
    return "\n".join(wanted) + ("\n" if wanted else "")


def import_lines(
    set_id: str, text: str, *, naming_dir: Path | None = None
) -> tuple[int, int]:
    """Append returned records. Returns ``(added, skipped)``.

    Skips anything already present rather than trusting the sender: a reader who
    submits twice must not double their weight in the statistic.
    """
    existing = {json.dumps(r, sort_keys=True) for r in _records(set_id, naming_dir)}
    added = skipped = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if not isinstance(data, dict) or data.get("kind") not in {"name", "judgement"}:
            skipped += 1
            continue
        record = {str(k): str(v) for k, v in data.items()}
        if json.dumps(record, sort_keys=True) in existing:
            skipped += 1
            continue
        _append(set_id, record, naming_dir)
        existing.add(json.dumps(record, sort_keys=True))
        added += 1
    return added, skipped


def load_set(set_id: str, *, naming_dir: Path | None = None) -> NamingSet | None:
    path = _set_path(set_id, naming_dir)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    clusters = tuple(
        FrozenCluster(
            cluster_id=str(c.get("cluster_id", "")),
            paper_ids=tuple(str(p) for p in c.get("paper_ids", [])),
            computed_label=tuple(str(w) for w in c.get("computed_label", [])),
        )
        for c in data.get("clusters", [])
        if isinstance(c, dict)
    )
    return NamingSet(
        set_id=str(data.get("set_id", set_id)),
        domain=str(data.get("domain", "")),
        created_at=str(data.get("created_at", "")),
        clusters=clusters,
    )


def list_sets(*, naming_dir: Path | None = None) -> tuple[NamingSet, ...]:
    root = _root(naming_dir)
    if not root.exists():
        return ()
    found = [load_set(p.name[: -len(".set.json")], naming_dir=naming_dir)
             for p in sorted(root.glob("*.set.json"))]
    return tuple(s for s in found if s is not None)


def _append(set_id: str, payload: Mapping[str, str], naming_dir: Path | None) -> None:
    path = _log_path(set_id, naming_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _records(set_id: str, naming_dir: Path | None) -> list[dict[str, str]]:
    path = _log_path(set_id, naming_dir)
    if not path.exists():
        return []
    out: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue  # a truncated write must not lose the rest of the log
        if isinstance(data, dict) and data.get("kind"):
            out.append({str(k): str(v) for k, v in data.items()})
    return out


def names(set_id: str, *, naming_dir: Path | None = None) -> tuple[Name, ...]:
    """Every name written for this set, oldest first."""
    return tuple(
        Name(
            set_id=set_id,
            cluster_id=r.get("cluster_id", ""),
            coder=r.get("coder", ""),
            text=r.get("text", ""),
            written_at=r.get("at", ""),
            source=r.get("source", HUMAN),
        )
        for r in _records(set_id, naming_dir)
        if r.get("kind") == "name"
    )


def judgements(set_id: str, *, naming_dir: Path | None = None) -> tuple[Judgement, ...]:
    return tuple(
        Judgement(
            set_id=set_id,
            cluster_id=r.get("cluster_id", ""),
            name_id=r.get("name_id", ""),
            judge=r.get("judge", ""),
            verdict=r.get("verdict", ""),
            judged_at=r.get("at", ""),
        )
        for r in _records(set_id, naming_dir)
        if r.get("kind") == "judgement"
    )


def to_name(
    naming_set: NamingSet, *, coder: str | None = None, naming_dir: Path | None = None
) -> tuple[FrozenCluster, ...]:
    """Clusters this person has not yet named, in snapshot order.

    Returns no one else's names, by construction rather than by discipline: this
    is the pass that has to stay blind.
    """
    who = coder or current_coder()
    done = {
        n.cluster_id
        for n in names(naming_set.set_id, naming_dir=naming_dir)
        if n.coder == who
    }
    return tuple(c for c in naming_set.clusters if c.cluster_id not in done)


def write_name(
    naming_set: NamingSet,
    cluster_id: str,
    text: str,
    *,
    coder: str | None = None,
    source: str = HUMAN,
    naming_dir: Path | None = None,
    now: dt.datetime | None = None,
) -> Name:
    """Record one name. Raises if the cluster is not in this set."""
    if naming_set.cluster(cluster_id) is None:
        raise ValueError(f"cluster {cluster_id!r} is not in set {naming_set.set_id!r}")
    written = visible(text)
    if not written:
        raise ValueError("a name cannot be empty")
    name = Name(
        set_id=naming_set.set_id,
        cluster_id=cluster_id,
        coder=coder or current_coder(),
        text=written,
        written_at=(now or dt.datetime.now(dt.UTC)).isoformat(),
        source=source,
    )
    _append(
        naming_set.set_id,
        {
            "kind": "name",
            "cluster_id": name.cluster_id,
            "coder": name.coder,
            "text": name.text,
            "at": name.written_at,
            "source": name.source,
        },
        naming_dir,
    )
    return name


def to_judge(
    naming_set: NamingSet, *, judge: str | None = None, naming_dir: Path | None = None
) -> tuple[Candidate, ...]:
    """Names this person has not yet ruled on, shuffled and unattributed.

    Including their own: a judge who only rules on other people's names knows
    which is which by elimination. The shuffle is seeded per judge, so the order
    carries no information but a half-finished pass can be resumed.
    """
    who = judge or current_coder()
    written = names(naming_set.set_id, naming_dir=naming_dir)
    ruled = {
        j.name_id
        for j in judgements(naming_set.set_id, naming_dir=naming_dir)
        if j.judge == who
    }
    pending = [
        Candidate(name_id=n.name_id, cluster_id=n.cluster_id, text=n.text)
        for n in written
        if n.name_id not in ruled
    ]
    random.Random(f"{naming_set.set_id}:{who}").shuffle(pending)
    return tuple(pending)


def write_judgement(
    naming_set: NamingSet,
    candidate: Candidate,
    verdict: str,
    *,
    judge: str | None = None,
    naming_dir: Path | None = None,
    now: dt.datetime | None = None,
) -> Judgement:
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}, got {verdict!r}")
    ruling = Judgement(
        set_id=naming_set.set_id,
        cluster_id=candidate.cluster_id,
        name_id=candidate.name_id,
        judge=judge or current_coder(),
        verdict=verdict,
        judged_at=(now or dt.datetime.now(dt.UTC)).isoformat(),
    )
    _append(
        naming_set.set_id,
        {
            "kind": "judgement",
            "cluster_id": ruling.cluster_id,
            "name_id": ruling.name_id,
            "judge": ruling.judge,
            "verdict": ruling.verdict,
            "at": ruling.judged_at,
        },
        naming_dir,
    )
    return ruling


def agreement(set_id: str, *, naming_dir: Path | None = None) -> Agreement:
    """Krippendorff's alpha over the rulings — the answer stage 1 exists to get."""
    ruled = judgements(set_id, naming_dir=naming_dir)
    # Last ruling wins: a judge who revisits a name has changed their mind, and
    # counting both would inflate the disagreement they just resolved.
    latest: dict[str, dict[str, str]] = {}
    for ruling in ruled:
        latest.setdefault(ruling.name_id, {})[ruling.judge] = ruling.verdict
    spread: dict[str, int] = {}
    for votes in latest.values():
        for verdict in votes.values():
            spread[verdict] = spread.get(verdict, 0) + 1
    shared = {unit: votes for unit, votes in latest.items() if len(votes) >= 2}
    return Agreement(
        alpha=krippendorff_alpha(latest),
        judges=len({r.judge for r in ruled}),
        units=len(shared),
        single_judged=len(latest) - len(shared),
        verdict_spread=dict(sorted(spread.items())),
    )
