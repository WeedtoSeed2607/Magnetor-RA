"""The naming experiment's screens, shared by the local dashboard and the public app.

Two entry points render this: the full dashboard, which reads titles from the
domain store, and :mod:`naming_app`, which reads them from a bundle and has no
corpus at all. Both call the same functions, so a fix to the reader's experience
cannot land in one and miss the other — which is exactly how the truncated
paper list survived as long as it did.

The title lookup is a parameter for that reason: it is the only thing the two
entry points genuinely disagree about.
"""

from __future__ import annotations

from collections.abc import Mapping

import streamlit as st

from magnetor.coding import ALPHA_FIRM, ALPHA_FLOOR
from magnetor.naming import (
    ACCEPTS,
    PARTIAL,
    REJECTS,
    VERDICT_PROMPT,
    NamingSet,
    agreement,
    to_judge,
    to_name,
    visible,
    write_judgement,
    write_name,
)

INTRO = (
    "Two people read the same groups of papers and say what each one is about. "
    "If you agree with each other, we have something to measure a machine "
    "against. If you do not, there is nothing for it to be right about."
)
SUBTITLE = (
    "It takes about twenty minutes. You will not see the other person's answers "
    "until everyone has finished writing."
)


def papers_box(
    paper_ids: tuple[str, ...], titles: Mapping[str, str], *, height: int = 320
) -> None:
    """Every paper in the group, scrollable.

    All of them, never a sample: a reader shown ten of thirty-five names those
    ten, and the two readers would then be agreeing about an excerpt. Ordered
    nearest the group's centre first, so the most representative are read before
    the scrolling starts.
    """
    st.caption(f"All {len(paper_ids)} papers in this group — scroll to read them.")
    with st.container(height=height):
        for pid in paper_ids:
            st.markdown(f"- {titles.get(pid, pid)}")


def write_step(naming_set: NamingSet, titles: Mapping[str, str], who: str) -> None:
    pending = to_name(naming_set, coder=who)
    total = len(naming_set.clusters)
    done = total - len(pending)
    st.progress(done / total if total else 0.0, text=f"{done} of {total} named")
    if not pending:
        st.success(
            "You have named every group. Move on to step 2 once the other reader "
            "has finished writing theirs."
        )
        return
    cluster = pending[0]
    st.caption(
        "What are these papers about? One short line. There is no right wording — "
        "write what you would tell a colleague."
    )
    papers_box(cluster.paper_ids, titles)
    with st.form(f"name_{cluster.cluster_id}", clear_on_submit=True):
        answer = st.text_input("This group is about…")
        if st.form_submit_button("Save and show the next group"):
            if not visible(answer):
                st.warning("Write something first.")
            else:
                write_name(naming_set, cluster.cluster_id, answer, coder=who)
                st.rerun()


def judge_step(naming_set: NamingSet, titles: Mapping[str, str], who: str) -> None:
    pending = to_judge(naming_set, judge=who)
    if not pending:
        st.success("Nothing left to rule on. See step 3 for the result.")
        return
    st.caption(
        f"{len(pending)} name(s) to rule on. Your own are mixed in and the order is "
        "shuffled, so do not try to work out who wrote what — just answer."
    )
    candidate = pending[0]
    cluster = naming_set.cluster(candidate.cluster_id)
    if cluster is not None:
        papers_box(cluster.paper_ids, titles, height=240)
    st.markdown(f"### {candidate.text}")
    st.caption("Does that name what this group is about?")
    columns = st.columns(3)
    for column, verdict in zip(columns, (ACCEPTS, PARTIAL, REJECTS), strict=True):
        if column.button(
            VERDICT_PROMPT[verdict], key=f"{candidate.name_id}_{verdict}"
        ):
            write_judgement(naming_set, candidate, verdict, judge=who)
            st.rerun()


def result_step(naming_set: NamingSet) -> None:
    result = agreement(naming_set.set_id)
    st.markdown(f"**{result.summary()}**")
    if result.measurable and result.alpha is not None:
        if result.alpha >= ALPHA_FIRM:
            st.success(
                "The two readers agree strongly. Naming a group is a real task with "
                "a right answer, so a machine's attempt at it can be scored."
            )
        elif result.alpha >= ALPHA_FLOOR:
            st.info(
                "The two readers mostly agree. Good enough to compare a machine "
                "against, not good enough to treat any single name as correct."
            )
        else:
            st.warning(
                "The two readers do not agree. That is a finding, not a failure: if "
                "people cannot say what these groups are about, there is nothing for "
                "a machine to be right about, and the question needs sharpening "
                "before any model is worth trying."
            )
    if result.single_judged:
        st.caption(
            f"{result.single_judged} name(s) have only one ruling so far and cannot "
            "count towards agreement."
        )
    if result.verdict_spread:
        st.caption(
            "Rulings so far: "
            + ", ".join(f"{k} {v}" for k, v in result.verdict_spread.items())
        )
