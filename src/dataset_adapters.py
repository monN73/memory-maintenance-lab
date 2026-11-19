"""Dataset adapters normalizing LongMemEval, scenario, and custom datasets into a unified turns + probes schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class UnifiedTurn:
    turn_id: int
    session_id: str
    timestamp: str
    user_text: str
    assistant_text: str | None = None
    source_message_indices: list[int] = field(default_factory=list)


@dataclass
class UnifiedProbe:
    probe_id: str
    question: str
    expected: str
    question_type: str = "general"
    is_abstention: bool = False
    active_from_turn: int = 1
    expected_by_turn: dict[str, str] = field(default_factory=dict)


@dataclass
class UnifiedItem:
    item_id: str
    dataset: str
    question_type: str
    is_abstention: bool
    question: str
    answer: str
    question_date: str | None
    turns: list[UnifiedTurn]
    probes: list[UnifiedProbe]


def align_longmemeval_turns(item: dict[str, Any]) -> list[UnifiedTurn]:
    """Convert LongMemEval haystack sessions into sequential 1-indexed UnifiedTurn objects."""
    turns: list[UnifiedTurn] = []
    global_turn = 0
    msg_idx = 0
    for sid, s_date, sess in zip(
        item["haystack_session_ids"],
        item["haystack_dates"],
        item["haystack_sessions"],
    ):
        i = 0
        while i < len(sess):
            m = sess[i]
            if m["role"] == "user":
                u_text = m["content"]
                u_idx = msg_idx
                msg_idx += 1
                a_text = None
                indices = [u_idx]
                if i + 1 < len(sess) and sess[i + 1]["role"] == "assistant":
                    a_text = sess[i + 1]["content"]
                    indices.append(msg_idx)
                    msg_idx += 1
                    i += 2
                else:
                    i += 1
                global_turn += 1
                turns.append(
                    UnifiedTurn(
                        turn_id=global_turn,
                        session_id=sid,
                        timestamp=s_date,
                        user_text=u_text,
                        assistant_text=a_text,
                        source_message_indices=indices,
                    )
                )
            else:
                global_turn += 1
                turns.append(
                    UnifiedTurn(
                        turn_id=global_turn,
                        session_id=sid,
                        timestamp=s_date,
                        user_text="",
                        assistant_text=m["content"],
                        source_message_indices=[msg_idx],
                    )
                )
                msg_idx += 1
                i += 1
    return turns


def longmemeval_active_from_turn(item: dict[str, Any], turns: list[UnifiedTurn]) -> int:
    """Return the first turn_id after all answer_session_ids have been ingested."""
    answer_sids = set(item.get("answer_session_ids") or [])
    if not answer_sids or not turns:
        return 1
    evidence_turns = [t.turn_id for t in turns if t.session_id in answer_sids]
    return max(evidence_turns) if evidence_turns else 1


def adapt_longmemeval_item(item: dict[str, Any], dataset_name: str = "LongMemEval") -> UnifiedItem:
    """Adapt a LongMemEval record into UnifiedItem with an evidence-aligned probe."""
    turns = align_longmemeval_turns(item)
    qid = item["question_id"]
    qtype = item.get("question_type", "general")
    is_abs = "_abs" in qid
    active_from = longmemeval_active_from_turn(item, turns)
    probe = UnifiedProbe(
        probe_id=f"{qid}-main",
        question=item["question"],
        expected=item["answer"],
        question_type=qtype,
        is_abstention=is_abs,
        active_from_turn=active_from,
    )
    return UnifiedItem(
        item_id=qid,
        dataset=dataset_name,
        question_type=qtype,
        is_abstention=is_abs,
        question=item["question"],
        answer=item["answer"],
        question_date=item.get("question_date"),
        turns=turns,
        probes=[probe],
    )


def adapt_unified_record(raw: dict[str, Any]) -> UnifiedItem:
    """Load an already-normalized dataset record (e.g. converted LoCoMo / GoodAI / custom)."""
    turns = [
        UnifiedTurn(
            turn_id=int(t.get("turn_id", idx + 1)),
            session_id=str(t.get("session_id", "s1")),
            timestamp=str(t.get("timestamp", "")),
            user_text=str(t.get("user_text", t.get("user", ""))),
            assistant_text=t.get("assistant_text", t.get("assistant")),
            source_message_indices=list(t.get("source_message_indices", [])),
        )
        for idx, t in enumerate(raw.get("turns", []))
    ]
    probes = [
        UnifiedProbe(
            probe_id=str(p.get("probe_id", f"p{idx + 1}")),
            question=str(p["question"]),
            expected=str(p.get("expected", "")),
            question_type=str(p.get("question_type", raw.get("question_type", "general"))),
            is_abstention=bool(p.get("is_abstention", raw.get("is_abstention", False))),
            active_from_turn=int(p.get("active_from_turn", 1)),
            expected_by_turn=dict(p.get("expected_by_turn", {})),
        )
        for idx, p in enumerate(raw.get("probes", []))
    ]
    if not probes and "question" in raw and "answer" in raw:
        probes.append(
            UnifiedProbe(
                probe_id=f"{raw['item_id']}-main",
                question=str(raw["question"]),
                expected=str(raw["answer"]),
                question_type=str(raw.get("question_type", "general")),
                is_abstention=bool(raw.get("is_abstention", False)),
                active_from_turn=int(raw.get("active_from_turn", 1)),
            )
        )
    return UnifiedItem(
        item_id=str(raw["item_id"]),
        dataset=str(raw.get("dataset", "unified")),
        question_type=str(raw.get("question_type", "general")),
        is_abstention=bool(raw.get("is_abstention", False)),
        question=str(raw.get("question", probes[0].question if probes else "")),
        answer=str(raw.get("answer", probes[0].expected if probes else "")),
        question_date=raw.get("question_date"),
        turns=turns,
        probes=probes,
    )
