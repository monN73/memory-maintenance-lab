#!/usr/bin/env python3
"""Generate a native, editable academic-style PowerPoint presentation (.pptx)
from the verified LongMemEval N0-N14 memory maintenance experiment data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["hatch.linewidth"] = 0.55

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parent.parent
OUT_PPTX = ROOT / "docs" / "memory_maintenance_report.pptx"
CHART_DIR = ROOT / "docs" / "_ppt_charts"

# Academic color palette
NAVY = RGBColor(15, 41, 66)       # #0F2942
BLUE = RGBColor(31, 90, 138)      # #1F5A8A
TEAL = RGBColor(14, 124, 123)     # #0E7C7B
CORAL = RGBColor(192, 57, 43)     # #C0392B
AMBER = RGBColor(217, 119, 6)     # #D97706
PURPLE = RGBColor(109, 40, 217)   # #6D28D9
SLATE = RGBColor(71, 85, 105)     # #475569
DARK = RGBColor(30, 41, 59)       # #1E293B
MUTED = RGBColor(100, 116, 139)   # #64748B
BORDER = RGBColor(203, 213, 225)  # #CBD5E1
CARD_BG = RGBColor(248, 250, 252) # #F8FAFC
SOFT_BLUE = RGBColor(239, 246, 255)
SOFT_TEAL = RGBColor(240, 253, 250)
SOFT_RED = RGBColor(254, 242, 242)
SOFT_AMBER = RGBColor(255, 251, 235)
WHITE = RGBColor(255, 255, 255)

FONT_CN = "Microsoft YaHei"


def _load_overview() -> dict[str, Any]:
    raw = (ROOT / "showcase" / "data" / "overview.js").read_text(encoding="utf-8")
    prefix = "window.__SHOWCASE_OVERVIEW__ = "
    idx = raw.find(prefix)
    if idx < 0:
        raise RuntimeError("Unexpected overview.js format")
    payload = raw[idx + len(prefix) :].strip()
    if payload.endswith(";"):
        payload = payload[:-1]
    return json.loads(payload)


def _set_font(run, size: int, bold: bool = False, color: RGBColor = DARK, name: str = FONT_CN):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color


def _add_header(slide, title: str, subtitle: str, page_num: int):
    # Top thin accent bar
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.08))
    bar.fill.solid()
    bar.fill.fore_color.rgb = BLUE
    bar.line.fill.background()

    # Title box
    tb = slide.shapes.add_textbox(Inches(0.55), Inches(0.22), Inches(11.6), Inches(0.55))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = title
    _set_font(r, 23, bold=True, color=NAVY)

    # Subtitle box
    sb = slide.shapes.add_textbox(Inches(0.55), Inches(0.75), Inches(11.6), Inches(0.32))
    stf = sb.text_frame
    stf.word_wrap = True
    stf.margin_left = stf.margin_top = stf.margin_right = stf.margin_bottom = 0
    sp = stf.paragraphs[0]
    sr = sp.add_run()
    sr.text = subtitle
    _set_font(sr, 12, bold=False, color=MUTED)

    # Divider line
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(1.12), Inches(12.23), Inches(0.015))
    line.fill.solid()
    line.fill.fore_color.rgb = BORDER
    line.line.fill.background()

    # Footer
    ft = slide.shapes.add_textbox(Inches(0.55), Inches(7.12), Inches(10.5), Inches(0.28))
    ftf = ft.text_frame
    ftf.margin_left = ftf.margin_top = ftf.margin_right = ftf.margin_bottom = 0
    fp = ftf.paragraphs[0]
    fr = fp.add_run()
    fr.text = "Memory Maintenance Lab · LongMemEval (N0–N14, B=8192, ecnu-plus)"
    _set_font(fr, 9, color=MUTED)

    pg = slide.shapes.add_textbox(Inches(12.0), Inches(7.12), Inches(0.78), Inches(0.28))
    ptf = pg.text_frame
    ptf.margin_left = ptf.margin_top = ptf.margin_right = ptf.margin_bottom = 0
    pp = ptf.paragraphs[0]
    pp.alignment = PP_ALIGN.RIGHT
    pr = pp.add_run()
    pr.text = f"{page_num} / 11"
    _set_font(pr, 9, bold=True, color=MUTED)


def _add_card(
    slide,
    left: float,
    top: float,
    width: float,
    height: float,
    title: str,
    bullets: list[tuple[str, str]],
    bg: RGBColor = CARD_BG,
    accent: RGBColor = BLUE,
    title_size: int = 14,
    body_size: int = 11,
):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    card.adjustments[0] = 0.03
    card.fill.solid()
    card.fill.fore_color.rgb = bg
    card.line.color.rgb = BORDER
    card.line.width = Pt(1)

    # Top accent strip inside card
    strip = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(0.06))
    strip.fill.solid()
    strip.fill.fore_color.rgb = accent
    strip.line.fill.background()

    tb = slide.shapes.add_textbox(Inches(left + 0.16), Inches(top + 0.12), Inches(width - 0.32), Inches(height - 0.2))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p0 = tf.paragraphs[0]
    p0.space_after = Pt(5)
    r0 = p0.add_run()
    r0.text = title
    _set_font(r0, title_size, bold=True, color=NAVY)

    for lead, text in bullets:
        p = tf.add_paragraph()
        p.space_after = Pt(3.5)
        if lead:
            rl = p.add_run()
            rl.text = lead + " "
            _set_font(rl, body_size, bold=True, color=accent)
        rt = p.add_run()
        rt.text = text
        _set_font(rt, body_size, bold=False, color=DARK)
    return card


def _make_plan_verification_chart(pa: dict[str, Any], pb: dict[str, Any], out_path: Path) -> Path:
    """High-DPI Matplotlib two-panel horizontal bar chart for Plan A & Plan B verification."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(5.8, 5.2), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    panels = [
        (axes[0], f"Plan A: s-ku-10-v1 Tail Replay ({pa['total']:,} cells)", pa),
        (axes[1], f"Plan B: s-dev-30 Checkpoints ({pb['total']:,} cells)", pb),
    ]
    labels = ["Unexpected Missing\n/ Mismatch", "Legal Termination\nGaps", "Completed Valid\nProbes"]
    colors = ["#D98880", "#F3C677", "#3B6E8C"]

    for ax, title, d in panels:
        vals = [
            int(d["unexpected_missing"]) + int(d["mismatch_error"]),
            int(d["termination_gaps"]),
            int(d["reusable_ok"]),
        ]
        total = max(1, int(d["total"]))
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, vals, height=0.54, color=colors, edgecolor="#334155", linewidth=0.6, zorder=3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8.5, color="#334155", fontweight="bold")
        ax.set_xlim(0, max(vals) * 1.28)
        ax.grid(True, axis="x", linestyle="--", alpha=0.32, color="#94A3B8", zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CBD5E1")
        ax.spines["bottom"].set_color("#CBD5E1")
        ax.tick_params(axis="x", labelsize=8, colors="#475569")
        ax.set_title(title, fontsize=9.8, fontweight="bold", color="#0F2942", loc="left", pad=8)

        for bar, v in zip(bars, vals):
            pct = 100.0 * v / total
            ax.text(
                v + max(vals) * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{v:,} ({pct:.1f}%)",
                va="center",
                ha="left",
                fontsize=8.5,
                fontweight="bold",
                color="#1E293B",
            )

    plt.tight_layout(pad=1.3)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _make_accuracy_chart(strategies: list[dict[str, Any]], out_path: Path) -> Path:
    """High-DPI Matplotlib grouped bar chart for N0-N14 across 3 benchmarks."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.2, 3.85), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    sids = [s["id"] for s in strategies]
    x = np.arange(len(sids))
    w = 0.26

    dev_vals = [round(float(s["s_dev_30"]["end_to_end_success"]) * 100, 1) if s.get("s_dev_30") else 0.0 for s in strategies]
    ku_vals = [round(float(s["s_ku_10"]["end_to_end_success"]) * 100, 1) if s.get("s_ku_10") else 0.0 for s in strategies]
    o5_vals = [round(float(s["oracle_5"]["end_to_end_success"]) * 100, 1) if s.get("oracle_5") else 0.0 for s in strategies]

    b1 = ax.bar(x - w, dev_vals, width=w, label="s-dev-30 (n=30)", color="#3B6E8C", edgecolor="#1E293B", linewidth=0.55, zorder=3)
    b2 = ax.bar(x, ku_vals, width=w, label="s-ku-10 (n=10)", color="#99C7B8", edgecolor="#1E293B", linewidth=0.55, zorder=3)
    b3 = ax.bar(x + w, o5_vals, width=w, label="oracle-5 (n=5)", color="#E2E8F0", edgecolor="#334155", linewidth=0.55, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(sids, fontsize=9.5, fontweight="bold", color="#0F2942")
    ax.set_ylim(0, 114)
    ax.set_ylabel("End-to-End Accuracy (%)", fontsize=9.5, fontweight="bold", color="#1E293B")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35, color="#94A3B8", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E1")
    ax.spines["bottom"].set_color("#CBD5E1")
    ax.tick_params(axis="y", labelsize=8.5, colors="#475569")
    ax.legend(loc="upper right", ncol=3, frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=8.8)

    # Annotate bar values cleanly
    for idx, (r1, r2, r3) in enumerate(zip(b1, b2, b3)):
        for bar, val, is_zero_note in [
            (r1, dev_vals[idx], sids[idx] == "N1"),
            (r2, ku_vals[idx], sids[idx] == "N0"),
            (r3, o5_vals[idx], False),
        ]:
            lbl = "0%*" if is_zero_note else (f"{val:.0f}" if val in (0.0, 20.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0) else f"{val:.1f}")
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 1.5,
                lbl,
                ha="center",
                va="bottom",
                fontsize=6.8,
                fontweight="bold" if val >= 60 or is_zero_note else "normal",
                color="#C0392B" if is_zero_note else "#334155",
            )

    ax.text(
        0.005, 0.94,
        "* N0 (s-ku-10) = 0% (0/10, no memory); N1 (s-dev-30) = 0% (29/30 context overflow)",
        transform=ax.transAxes, fontsize=7.8, color="#64748B", style="italic",
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _make_pareto_chart(strategies: list[dict[str, Any]], out_path: Path) -> Path:
    """High-DPI Matplotlib scatter plot: s-dev-30 Accuracy vs Total Tokens (log scale)."""
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.3), dpi=300)

    family_colors = {
        "baseline": "#64748B",
        "window_compact": "#C0392B",
        "archive_bm25": "#0E7C7B",
        "fact_hybrid": "#1F5A8A",
    }
    family_labels = {
        "baseline": "Baseline (N0, N1, N5)",
        "window_compact": "Window / LLM Compaction (N2-N4, N6)",
        "archive_bm25": "External BM25 Archive (N7, N8, N11, N12)",
        "fact_hybrid": "Fact Table / Hybrid (N9, N10, N13, N14)",
    }
    plotted_families = set()

    # Label offsets to avoid overlap around 10^5 tokens
    offsets = {
        "N0": (8, 4),
        "N1": (-58, -6),
        "N2": (16, -6),
        "N3": (18, 2),
        "N4": (-28, -14),
        "N5": (-58, 4),
        "N6": (-28, 8),
        "N7": (-22, -15),
        "N8": (-75, 16),
        "N9": (-82, -8),
        "N10": (18, 2),
        "N11": (-82, 4),
        "N12": (18, 18),
        "N13": (-75, 28),
        "N14": (18, 8),
    }

    for s in strategies:
        d30 = s.get("s_dev_30")
        if not d30:
            continue
        sid = s["id"]
        fam = s["family"]
        tok = max(1, int(d30["total_tokens"]))
        acc = float(d30["end_to_end_success"]) * 100.0
        lbl = family_labels[fam] if fam not in plotted_families else None
        plotted_families.add(fam)
        ax.scatter(
            tok, acc,
            s=95 if sid in ("N7", "N6", "N4", "N1") else 65,
            color=family_colors[fam],
            edgecolors="white",
            linewidth=1.0,
            zorder=4,
            label=lbl,
        )
        dx, dy = offsets.get(sid, (6, 4))
        ax.annotate(
            f"{sid} ({acc:.1f}%)",
            (tok, acc),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.0,
            fontweight="bold" if sid in ("N7", "N6", "N4") else "normal",
            color="#1E293B",
            arrowprops={"arrowstyle": "-", "color": "#94A3B8", "lw": 0.6} if sid not in ("N0", "N4", "N6", "N7") else None,
        )

    ax.set_xscale("log")
    ax.set_xlim(3500, 22000000)
    ax.set_ylim(-4, 68)
    ax.grid(True, which="major", linestyle="--", alpha=0.35, color="#94A3B8")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("s-dev-30 Total Tokens (Log Scale)", fontsize=9.5, fontweight="bold", color="#1E293B")
    ax.set_ylabel("End-to-End Accuracy (%)", fontsize=9.5, fontweight="bold", color="#1E293B")
    ax.set_title("Accuracy vs. Token Cost Trade-off (s-dev-30, n=30)", fontsize=10.5, fontweight="bold", color="#0F2942", pad=10)
    ax.legend(loc="upper right", fontsize=7.5, frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def build_pptx():
    ov = _load_overview()
    strategies = ov["strategies"]
    sdev = ov["benchmarks"]["s_dev_30"]
    sku = ov["benchmarks"]["s_ku_10"]
    pa = ov["plans"]["plan_a"]["checklist"]
    pb = ov["plans"]["plan_b"]["checklist"]

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # ---------------------------------------------------------
    # SLIDE 1: Cover
    # ---------------------------------------------------------
    s1 = prs.slides.add_slide(blank)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = NAVY
    bg1.line.fill.background()

    stripe = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(0.28), Inches(7.5))
    stripe.fill.solid()
    stripe.fill.fore_color.rgb = TEAL
    stripe.line.fill.background()

    tb = s1.shapes.add_textbox(Inches(0.95), Inches(1.55), Inches(11.4), Inches(2.2))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = "受控上下文预算下的长程对话记忆维护策略评测与失效归因"
    _set_font(r, 34, bold=True, color=WHITE)

    p2 = tf.add_paragraph()
    p2.space_before = Pt(16)
    r2 = p2.add_run()
    r2.text = "基于 LongMemEval 的 N0–N14（15 种策略）对比、离线冻结快照回放与反事实因果分析"
    _set_font(r2, 17, bold=False, color=RGBColor(203, 213, 225))

    # 4 KPI pills on cover
    cover_kpis = [
        ("15 种策略全集", "N0–N14（含 N6 分块摘要）"),
        ("450 + 150 终态格", "s-dev-30 (30题) + s-ku-10 (10题)"),
        ("4,950 + 552 探针", "Plan A / Plan B 冻结快照重放"),
        ("在线 Showcase 演示", "monn73.github.io/memory-maintenance-lab"),
    ]
    for i, (k_title, k_sub) in enumerate(cover_kpis):
        x = 0.95 + i * 2.92
        box = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(4.65), Inches(2.72), Inches(1.25))
        box.adjustments[0] = 0.06
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(23, 58, 94)
        box.line.color.rgb = TEAL
        box.line.width = Pt(1)
        ktf = box.text_frame
        ktf.word_wrap = True
        ktf.margin_left = Inches(0.14)
        ktf.margin_top = Inches(0.18)
        kp1 = ktf.paragraphs[0]
        kr1 = kp1.add_run()
        kr1.text = k_title
        _set_font(kr1, 15, bold=True, color=WHITE)
        kp2 = ktf.add_paragraph()
        kp2.space_before = Pt(6)
        kr2 = kp2.add_run()
        kr2.text = k_sub
        _set_font(kr2, 11, color=RGBColor(186, 230, 253))

    ft1 = s1.shapes.add_textbox(Inches(0.95), Inches(6.55), Inches(11.4), Inches(0.4))
    ftf1 = ft1.text_frame
    fp1 = ftf1.paragraphs[0]
    fr1 = fp1.add_run()
    fr1.text = "统一基准：LongMemEval-S  |  问答与裁判模型：ecnu-plus  |  受控预算：B = 8192 tokens（M ≤ 2048，归档 ≤ 16384）"
    _set_font(fr1, 12, color=RGBColor(148, 163, 184))

    # ---------------------------------------------------------
    # SLIDE 2: Memory Workflow & Role
    # ---------------------------------------------------------
    s2 = prs.slides.add_slide(blank)
    _add_header(
        s2,
        "Memory 的工作流程与核心作用",
        "在受控上下文预算（B=8192）内替代全量历史：防溢出、留关键事实、支持知识更新",
        2,
    )

    steps = [
        ("1. 对话写入", "Ingest", "每轮 User + Assistant 完整对话写入近期历史缓冲。", BLUE),
        ("2. 预算检测", "Budget Check", "统计可见历史与记忆槽开销；超过 trigger > 6553 时触发维护。", AMBER),
        ("3. 记忆维护", "Maintenance", "滑窗丢弃 / 抽取或 LLM 摘要 / 归档入库 / 事实表更新，压回 ≤ 4915。", CORAL),
        ("4. 读出组装", "Readout", "将保留的近期轮次与显式记忆槽 M（≤2048）组装为受控 Prompt。", TEAL),
        ("5. 问答推理", "Answering", "模型基于受控上下文生成答案，在预算安全与信息留存间权衡。", PURPLE),
    ]
    for i, (st_t, st_en, st_desc, col) in enumerate(steps):
        x = 0.55 + i * 2.50
        box = s2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(1.45), Inches(2.22), Inches(2.25))
        box.adjustments[0] = 0.06
        box.fill.solid()
        box.fill.fore_color.rgb = CARD_BG
        box.line.color.rgb = col
        box.line.width = Pt(1.5)
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.12)
        tf.margin_right = Inches(0.12)
        tf.margin_top = Inches(0.14)
        p0 = tf.paragraphs[0]
        r0 = p0.add_run()
        r0.text = st_t
        _set_font(r0, 14, bold=True, color=col)
        p1 = tf.add_paragraph()
        p1.space_before = Pt(2)
        r1 = p1.add_run()
        r1.text = st_en
        _set_font(r1, 10, bold=True, color=MUTED)
        p2 = tf.add_paragraph()
        p2.space_before = Pt(8)
        r2 = p2.add_run()
        r2.text = st_desc
        _set_font(r2, 11, color=DARK)

        if i < 4:
            arr = s2.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + 2.25), Inches(2.42), Inches(0.22), Inches(0.26))
            arr.fill.solid()
            arr.fill.fore_color.rgb = SLATE
            arr.line.fill.background()

    _add_card(
        s2, 0.55, 4.05, 5.95, 2.85,
        "为什么不能只靠“全量拼接”？（N1 的现实困境）",
        [
            ("• 短历史理想态：", "oracle-5（去噪短历史）下 N1 全量上下文可达 100.0% (5/5)。"),
            ("• 长历史崩盘：", "LongMemEval-S 单题常达 200–500+ 轮，N1 在 s-dev-30 上 29/30 直接上下文溢出（0.0%）。"),
            ("• 成本与干扰：", "全量输入不仅易超窗，还会让大量无关会话噪声稀释关键证据注意力。"),
        ],
        bg=SOFT_RED, accent=CORAL, title_size=14, body_size=11.5,
    )
    _add_card(
        s2, 6.83, 4.05, 5.95, 2.85,
        "Memory 机制在系统中的三大核心作用",
        [
            ("1. 预算守门：", "通过裁剪/压缩/外置，把每轮可见上下文稳定约束在 B=8192 以内，实现零溢出。"),
            ("2. 跨会话留存：", "将滑出窗口的早期事实通过摘要、BM25 归档或结构化事实表跨几百轮保留。"),
            ("3. 知识更新裁决：", "当用户偏好或状态发生变更时，区分最新状态与过时旧值，抑制旧值干扰。"),
        ],
        bg=SOFT_TEAL, accent=TEAL, title_size=14, body_size=11.5,
    )

    # ---------------------------------------------------------
    # SLIDE 3: Background & Three Contradictions
    # ---------------------------------------------------------
    s3 = prs.slides.add_slide(blank)
    _add_header(
        s3,
        "研究背景：受控预算下的三重核心矛盾",
        "长程对话中“存得下、找得准、算得稳”无法同时免费获得，必须评估不同机制的取舍边界",
        3,
    )

    contradictions = [
        (
            "矛盾一：容量限制 vs 信息完整性",
            BLUE,
            SOFT_BLUE,
            [
                ("• 直接淘汰（N2）：", "最省 Token，但早期事实一旦滑出窗口便永久丢失（s-dev-30 仅 6.7%）。"),
                ("• 有损压缩（N3/N4/N6）：", "试图用 ≤2048 tokens 概括全部历史，但数字、时间与长尾细节极易失真。"),
                ("• 有界归档（N8/N11/N12）：", "16K 归档在 200–500 轮超长对话下仍会淘汰早期证据。"),
            ],
        ),
        (
            "矛盾二：旧值干扰 vs 知识更新",
            TEAL,
            SOFT_TEAL,
            [
                ("• 偏好/状态多次变更：", "如 s-ku-10 切片中用户多次更新计数、时间或偏好。"),
                ("• 只追加的隐患（N9/N7）：", "旧事实仍在库中，BM25 top-k 漂移或同时召回新旧片段会导致回答过时旧值。"),
                ("• 规则更新的代价（N10/N14）：", "显式覆盖/撤回能清理旧值，但受限于规则抽取召回率。"),
            ],
        ),
        (
            "矛盾三：维护开销 vs 执行稳定性",
            CORAL,
            SOFT_RED,
            [
                ("• Token 成本悬殊：", "规则/BM25 策略全量仅 ~120K–146K tokens；LLM 摘要（N4/N6）高达 7.6M–9.6M。"),
                ("• 预算失控风险：", "随着历史累积，LLM 自由文本合并极易连续两次超 M≤2048 上限。"),
                ("• 严格契约暴露：", "实验禁止静默截断，N4（4/30）与 N6（10/30）直接暴露 maintenance_budget_error。"),
            ],
        ),
    ]
    for i, (ct, col, bg, buls) in enumerate(contradictions):
        _add_card(s3, 0.55 + i * 4.18, 1.4, 3.88, 5.4, ct, buls, bg=bg, accent=col, title_size=14, body_size=11.5)

    # ---------------------------------------------------------
    # SLIDE 4: Datasets & Unified Protocol
    # ---------------------------------------------------------
    s4 = prs.slides.add_slide(blank)
    _add_header(
        s4,
        "实验数据集与统一评测口径",
        "全部基于公开 LongMemEval-S 基准构建，统一问答与裁判模型为 ecnu-plus，零 scoring_pending",
        4,
    )

    datasets = [
        (
            "s-dev-30（分层主战场）",
            "30 题 × 15 策略 = 450 格",
            BLUE,
            [
                ("• 题型覆盖：", "single-session、multi-session、temporal-reasoning、knowledge-update、preference 及拒答题（_abs）。"),
                ("• 历史长度：", "通常 200–500+ 轮跨会话噪声对话。"),
                ("• 终态结算：", "378 结算 + 58 溢出（N1/N5）+ 14 预算错误（N4/N6），0 API 错误。"),
            ],
        ),
        (
            "s-ku-10-v1（知识更新专题）",
            "10 题 × 15 策略 = 150 格",
            TEAL,
            [
                ("• 专题聚焦：", "10 道 knowledge-update 题目，重点检验“新值覆盖旧值”与“尾段记忆保持”。"),
                ("• 全量快照：", "保存每轮 snapshots/turn_*.json，支撑 Plan A 逐轮重放。"),
                ("• 证据轨迹：", "支持消息级与会话级双分母证据留存归因。"),
            ],
        ),
        (
            "oracle-5（短历史对照基线）",
            "5 题 × 15 策略",
            PURPLE,
            [
                ("• 构造方式：", "仅保留包含答案的关键会话，去除无关噪声会话。"),
                ("• 核心作用：", "证明 N1/N6 在短历史下可达 100.0% (5/5)，量化长程噪声导致的退化落差。"),
                ("• 覆盖策略：", "N0–N14 全部 15 个策略。"),
            ],
        ),
    ]
    for i, (dt, dsub, col, buls) in enumerate(datasets):
        _add_card(s4, 0.55 + i * 4.18, 1.4, 3.88, 3.65, f"{dt}\n{dsub}", buls, bg=CARD_BG, accent=col, title_size=13, body_size=11)

    _add_card(
        s4, 0.55, 5.28, 12.23, 1.55,
        "统一受控预算契约与防作弊口径",
        [
            ("• 预算参数：", "受控策略（N2–N4, N6–N14）统一 B=8192，trigger=6553，target=4915，显式记忆槽 M≤2048，有界归档上限 16384；N1/N5 为 131072 模型上下文上限。"),
            ("• 透明声明：", "s-ku-10 中 01493427 与 45dc21b6 曾用于研发调试（开发暴露样本）；全量批跑已排查重跑，所有策略均无 maintenance_api_error 残留。"),
        ],
        bg=SOFT_AMBER, accent=AMBER, title_size=13, body_size=11,
    )

    # ---------------------------------------------------------
    # SLIDE 5: 15 Strategies Taxonomy
    # ---------------------------------------------------------
    s5 = prs.slides.add_slide(blank)
    _add_header(
        s5,
        "N0–N14（15 种策略）设计谱系与对照维度",
        "涵盖 4 大技术家族，并内置单因子消融与方法束对照，避免“无对照跑分”",
        5,
    )

    families = [
        (
            "家族 1：基线对照（Baseline）",
            SLATE,
            [
                ("N0 无记忆基线：", "仅输入当前轮，验证题目非纯常识可答。"),
                ("N1 全量长上下文：", "131K 上限直接拼接全部历史（无维护）。"),
                ("N5 分块全量重放：", "按块重组全历史，检验分块边界一致性。"),
            ],
        ),
        (
            "家族 2：滑窗与摘要压缩（Window / Compaction）",
            CORAL,
            [
                ("N2 纯滑窗淘汰：", "超阈值直接丢弃最旧轮次，8K 受控下界。"),
                ("N3 确定性抽取：", "无 LLM，句级去重摘录填入 M≤2048。"),
                ("N4 LLM 增量摘要：", "每轮用 LLM 将溢出历史增量合并入摘要。"),
                ("N6 分块 LLM 摘要：", "先分块摘要（chunk）再合并（merge）。"),
            ],
        ),
        (
            "家族 3：外部归档 BM25（Archive BM25）",
            TEAL,
            [
                ("N7 无界 BM25：", "全历史入本地归档，读出时 BM25 召回 top-k。"),
                ("N8 16K 有界（FIFO）：", "归档上限 16384，先进先出淘汰。"),
                ("N11 16K 有界（去重）：", "写入前去重再 FIFO（单因子对照 N8）。"),
                ("N12 16K 有界（均衡）：", "按会话配额均衡淘汰（对照 N8/N11）。"),
            ],
        ),
        (
            "家族 4：结构化事实表与混合读取（Fact / Hybrid）",
            BLUE,
            [
                ("N9 只追加事实表：", "规则抽取结构化事实，只追加不覆盖。"),
                ("N10 可更新/撤回：", "支持规则匹配更新与撤回旧值（对照 N9）。"),
                ("N13 事实 + BM25 混合：", "结构化事实优先，空余槽位用 BM25 兜底（对照 N10）。"),
                ("N14 增强提取事实表：", "增强前端规则抽取召回（单因子消融 N10 vs N14）。"),
            ],
        ),
    ]
    coords = [(0.55, 1.38), (6.80, 1.38), (0.55, 4.18), (6.80, 4.18)]
    for (ft, col, buls), (x, y) in zip(families, coords):
        _add_card(s5, x, y, 5.98, 2.62, ft, buls, bg=CARD_BG, accent=col, title_size=13.5, body_size=11)

    # ---------------------------------------------------------
    # SLIDE 6: Evaluation Protocol & Snapshot Replay
    # ---------------------------------------------------------
    s6 = prs.slides.add_slide(blank)
    _add_header(
        s6,
        "测试方案：终态评测 + 冻结快照重放 + 反事实因果归因",
        "不止步于“终态准确率”，而是精确锁定“哪一轮首次答错”以及“是不是维护动作导致的”",
        6,
    )

    _add_card(
        s6, 0.55, 1.38, 6.2, 2.55,
        "1. 离线冻结快照重放（Plan A & Plan B）",
        [
            ("• Plan A（s-ku-10-v1 全量尾段）：", f"每题 [last_ev, total_turns] 共 330 题-轮 × 15 策略 = {pa['total']:,} 格；完成 {pa['reusable_ok']:,} 探针 + {pa['termination_gaps']} 终止缺口核验（0 意外缺失）。"),
            ("• Plan B（s-dev-30 7 题反转专题）：", f"在 N6, N8–N14 共同离散检查点（{pb['total']} 格，{pb['reusable_ok']} 完成 + {pb['termination_gaps']} N6 预算终止缺口）同轴对比，消除在线探针触发频率偏差。"),
        ],
        bg=SOFT_BLUE, accent=BLUE, title_size=13.5, body_size=11,
    )
    _add_card(
        s6, 0.55, 4.12, 6.2, 2.72,
        "2. 双重边界校验（--recheck & --skip-maintenance）",
        [
            ("• --recheck 边界复测：", f"同一冻结输入重放一次检验读出稳定性：Plan A {pa['rechecks_consistent']}/{pa['rechecks_total']} 一致（{pa['rechecks_instability']} 例读出摇摆），Plan B {pb['rechecks_consistent']}/{pb['rechecks_total']} 一致（{pb['rechecks_instability']} 例摇摆）。"),
            ("• --skip-maintenance 反事实对照：", f"在首次转错轮次跳过本轮维护（≤8192 预算内）：Plan A {pa['cf_inbudget_ok']}/{pa['cf_total']} 命中“跳过即答对、原维护答错”，Plan B {pb['cf_inbudget_ok']}/{pb['cf_total']} 命中，直接建立因果证据。"),
        ],
        bg=SOFT_TEAL, accent=TEAL, title_size=13.5, body_size=11,
    )

    plan_png = _make_plan_verification_chart(pa, pb, CHART_DIR / "plan_verification.png")
    s6.shapes.add_picture(str(plan_png), Inches(7.00), Inches(1.38), width=Inches(5.78))

    # ---------------------------------------------------------
    # SLIDE 7: Full 15-Strategy Results (Native Chart)
    # ---------------------------------------------------------
    s7 = prs.slides.add_slide(blank)
    _add_header(
        s7,
        "全量实验结果总览：15 策略端到端准确率对比",
        "N7（无界 BM25）在长历史与知识更新下均显著领先；N11 在更新密集场景提升明显；N4/N6 受预算失败拖累",
        7,
    )

    acc_png = _make_accuracy_chart(strategies, CHART_DIR / "accuracy_15_strategies.png")
    s7.shapes.add_picture(str(acc_png), Inches(0.55), Inches(1.30), width=Inches(12.23))

    # 3 summary takeaway cards below chart
    _add_card(
        s7, 0.55, 5.32, 3.88, 1.62,
        "1. 外部归档 BM25 显著领先",
        [
            ("• N7 无界：", "s-dev-30 60.0% (18/30)、s-ku-10 80.0% (8/10)，全量最佳。"),
            ("• N11 去重 16K：", "s-ku-10 达 70.0%（优于 N8 60% 与 N12 30%）。"),
        ],
        bg=SOFT_TEAL, accent=TEAL, title_size=12.5, body_size=10.5,
    )
    _add_card(
        s7, 4.72, 5.32, 3.88, 1.62,
        "2. LLM 摘要条件准确率高但失败多",
        [
            ("• N6 分块摘要：", "s-dev-30 23.3% (条件 35.0%, 10 预算失败)；ku10 50.0% (条件 62.5%)。"),
            ("• N4 增量摘要：", "s-dev-30 20.0% (4 失败)；ku10 40.0% (3 失败)。"),
        ],
        bg=SOFT_RED, accent=CORAL, title_size=12.5, body_size=10.5,
    )
    _add_card(
        s7, 8.89, 5.32, 3.88, 1.62,
        "3. 混合与增强提取优于纯规则",
        [
            ("• s-dev-30：", "N13 混合与 N14 增强均达 16.7% (5/30)，高于 N9/N10 的 10.0% (3/30)。"),
            ("• N1 短/长对比：", "oracle-5 100% vs s-dev-30 0% (29 溢出)。"),
        ],
        bg=SOFT_BLUE, accent=BLUE, title_size=12.5, body_size=10.5,
    )

    # ---------------------------------------------------------
    # SLIDE 8: Cost vs Accuracy Pareto + Deep Analysis
    # ---------------------------------------------------------
    s8 = prs.slides.add_slide(blank)
    _add_header(
        s8,
        "成本—准确率权衡（Pareto 分析）与策略优势对比",
        "N7 以约 132K tokens 实现最高准确率；N4/N6 消耗 58–73 倍 Token 却仍因预算上限频繁失败",
        8,
    )

    pareto_png = _make_pareto_chart(strategies, CHART_DIR / "pareto_sdev30.png")
    s8.shapes.add_picture(str(pareto_png), Inches(0.55), Inches(1.38), width=Inches(6.55))

    # Native comparison table on right
    rows_tbl = [
        ("策略家族", "代表", "s-dev-30 Token", "核心优势", "主要局限"),
        ("无界 BM25", "N7", "131,706", "准确率最高 (60%/80%)、低成本", "存储随历史增长；top-k 排序漂移"),
        ("16K 有界 BM25", "N11 / N8", "120K", "严格控容量；N11 去重更新强 (70%)", "超长历史下仍会淘汰早期证据"),
        ("事实表 + 混合", "N13 / N14", "121K–144K", "显式可更新/撤回、可解释性强", "开放域纯规则召回仍依赖 BM25 兜底"),
        ("LLM 摘要压缩", "N6 / N4", "7.60M–9.58M", "条件准确率高于滑窗 (35%/62.5%)", "58–73× Token；极易触发预算失败"),
        ("滑窗 / 机械抽取", "N2 / N3", "138K–147K", "实现极简、零额外 API 依赖", "信息留存最弱 (6.7%–13.3%)"),
    ]
    tbl_shape = s8.shapes.add_table(len(rows_tbl), 5, Inches(7.25), Inches(1.42), Inches(5.53), Inches(5.30))
    tbl = tbl_shape.table
    col_widths = [Inches(1.15), Inches(0.75), Inches(0.95), Inches(1.34), Inches(1.34)]
    for ci, cw in enumerate(col_widths):
        tbl.columns[ci].width = cw

    for ri, rdata in enumerate(rows_tbl):
        for ci, val in enumerate(rdata):
            cell = tbl.cell(ri, ci)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            if ri == 0:
                cell.fill.fore_color.rgb = NAVY
            elif ri == 1:
                cell.fill.fore_color.rgb = SOFT_TEAL
            elif ri == 4:
                cell.fill.fore_color.rgb = SOFT_RED
            else:
                cell.fill.fore_color.rgb = CARD_BG if ri % 2 == 0 else WHITE
            p = cell.text_frame.paragraphs[0]
            p.margin_left = p.margin_right = 0
            r = p.add_run()
            r.text = val
            _set_font(
                r,
                9.5 if ri > 0 else 10,
                bold=(ri == 0 or ci == 0),
                color=WHITE if ri == 0 else DARK,
            )

    # ---------------------------------------------------------
    # SLIDE 9: Showcase Live Demo Cases
    # ---------------------------------------------------------
    s9 = prs.slides.add_slide(blank)
    _add_header(
        s9,
        "典型失效案例与网页 Showcase 现场演示导览",
        "以下案例可直接切换至在线 Showcase（monn73.github.io/memory-maintenance-lab）逐轮检视三线证据",
        9,
    )

    demo_cases = [
        (
            "案例 1｜维护直接挤出关键证据（因果实证）",
            TEAL,
            SOFT_TEAL,
            [
                ("• 89941a93 · N9 · T185（自行车 4→2）：", "跳过本轮维护（7,304 ≤ 8,192）保留 T155 四辆车清单答对 4；执行维护淘汰后仅剩两辆答错为 2。"),
                ("• 4fd1909e · N9 · T130（演唱会地点）：", "跳过维护（7,320 ≤ 8,192）答对 Xfinity Center；执行维护后检索 [120..124] 答未知。"),
            ],
        ),
        (
            "案例 2｜BM25 排序漂移（未丢数据但被挤出 top-k）",
            BLUE,
            SOFT_BLUE,
            [
                ("• e66b632c · N7 · T251→T252（5K 个人最佳）：", "T251 检索 [17, 19, 20] 找回 27:45；T252 因新轮 T246 入池挤出 T17，模型反算 26:30+1:30 误答 28:00。"),
                ("• 启示：", "无界归档虽然没丢原文，但检索重排波动仍会造成读出阶段依据失落。"),
            ],
        ),
        (
            "案例 3｜边界读出不稳定 & 拒答对照（_abs）",
            AMBER,
            SOFT_AMBER,
            [
                ("• 4d6b87c8 · N9/N10 · T232→T234：", "--recheck 检出同一冻结输入下在“当前 25”与“拟新增后 27”间摇摆（readout_instability）。"),
                ("• bc8a6e93_abs · N6, N8–N14：", "拒答题标准答案为“未提及”，遗忘历史后模型仍倾向拒答，故在回放中不发生遗忘反转。"),
            ],
        ),
    ]
    for i, (ct, col, bg, buls) in enumerate(demo_cases):
        _add_card(s9, 0.55, 1.38 + i * 1.84, 12.23, 1.66, ct, buls, bg=bg, accent=col, title_size=13.5, body_size=11)

    # ---------------------------------------------------------
    # SLIDE 10: Memory Failure vs Model Capability Boundary
    # ---------------------------------------------------------
    s10 = prs.slides.add_slide(blank)
    _add_header(
        s10,
        "失效边界划定：记忆系统的锅 vs 模型能力的锅",
        "精细归因必须将“记忆没留住”与“证据已经在上下文中但模型仍答错/控长失败”严格区分",
        10,
    )

    boundaries = [
        (
            "1. 证据仍在记忆中，但模型仍然答错",
            CORAL,
            SOFT_RED,
            [
                ("• 现象定位：", "在 Showcase 热力图中显式单列为「! 证据仍在但答错」。"),
                ("• 归因结论：", "关键对话已保留在近期窗口或被 BM25/事实表成功注入 M，但模型在多跳推理、时间换算、反事实干扰或长上下文注意力分配上仍答错。"),
                ("• 工程含义：", "此类错误无法通过优化 Memory 存储/检索解决，瓶颈在于基座模型推理上限。"),
            ],
        ),
        (
            "2. 同输入读出摇摆（readout_instability）",
            AMBER,
            SOFT_AMBER,
            [
                ("• 现象定位：", "边界 --recheck 使用完全相同的 request_sha256 重放，回答与裁判却翻转。"),
                ("• 实测数据：", "Plan A 检出 4/76 例、Plan B 检出 1/48 例（如 4d6b87c8 在 25 与 27 间摇摆）。"),
                ("• 工程含义：", "属于基座模型在边界歧义问题上的采样/判别不稳定性。"),
            ],
        ),
        (
            "3. 维护预算失控（maintenance_budget_error）",
            BLUE,
            SOFT_BLUE,
            [
                ("• 现象定位：", "N4/N6 两次提示软硬上限及 <previous_attempt> 后，输出仍 > 2048 tokens。"),
                ("• 双重属性：", "本质是模型在高密度信息下精确控长与信息取舍能力不足；同时也是实验拒绝静默截断、严格执行契约的刻意暴露。"),
                ("• 评测规范：", "必须同时报告“端到端成功率”与“条件准确率”。"),
            ],
        ),
    ]
    for i, (bt, col, bg, buls) in enumerate(boundaries):
        _add_card(s10, 0.55 + i * 4.18, 1.4, 3.88, 5.4, bt, buls, bg=bg, accent=col, title_size=13.5, body_size=11)

    # ---------------------------------------------------------
    # SLIDE 11: Architectural Implications, Memory Conflict & Future Work
    # ---------------------------------------------------------
    s11 = prs.slides.add_slide(blank)
    _add_header(
        s11,
        "未来架构启示、记忆冲突问题与下一步工作",
        "从“受控预算下怎么留住记忆”走向“三层混合架构下的跨层冲突与真值裁决”",
        11,
    )

    # Three-tier architecture diagram (native shapes)
    tiers = [
        ("工作记忆（近期滑窗）", "保留最近数轮完整 raw 对话，保障当前语境连贯，零成本。", BLUE, 1.42),
        ("底层长期情景库（本地 raw + BM25/混合索引）", "N7 已证数字/日期高度依赖原文回溯；本地保留 raw + 索引几乎零推理 Token。", TEAL, 2.55),
        ("高层语义记忆（云端/后台异步 Fact 表 + 主题编排）", "避开每轮同步重写（N4/N6），会话边界异步更新结构化 Fact，抑制旧值干扰。", PURPLE, 3.68),
    ]
    for t_title, t_desc, col, y in tiers:
        box = s11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.55), Inches(y), Inches(6.35), Inches(0.95))
        box.adjustments[0] = 0.08
        box.fill.solid()
        box.fill.fore_color.rgb = CARD_BG
        box.line.color.rgb = col
        box.line.width = Pt(1.5)
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.15)
        tf.margin_top = Inches(0.10)
        p0 = tf.paragraphs[0]
        r0 = p0.add_run()
        r0.text = t_title
        _set_font(r0, 12.5, bold=True, color=col)
        p1 = tf.add_paragraph()
        p1.space_before = Pt(3)
        r1 = p1.add_run()
        r1.text = t_desc
        _set_font(r1, 10.5, color=DARK)

    _add_card(
        s11, 0.55, 4.88, 6.35, 1.98,
        "下一步工程落地计划",
        [
            ("1. 更大规模基准扩跑：", "在未开发暴露的 LongMemEval-S/M 长历史切片上扩跑核心对比策略。"),
            ("2. 真实 Agent 链路验证：", "将混合记忆策略接入 Hermes 等真实 Agent 工具调用链，检验复杂交互下的迁移性。"),
        ],
        bg=SOFT_BLUE, accent=BLUE, title_size=13, body_size=10.5,
    )

    _add_card(
        s11, 7.18, 1.42, 5.60, 5.44,
        "核心开放问题：跨层记忆冲突（Memory Conflict）",
        [
            ("• 为什么必然出现冲突？", "一旦同时保留“本地永久 raw”与“云端异步 Fact 表”，新旧陈述与跨层版本差将不可避免。"),
            ("1. 新旧事实冲突：", "BM25 同时召回历史旧状态原文与最新修改原文，模型易受旧值干扰。"),
            ("2. 跨层版本差：", "异步编排尚未完成时，近期滑窗/本地 raw 已更新，而云端 Fact 仍为旧版。"),
            ("3. 结构化 vs 原文冲突：", "N13 混合读取中，当 Fact 提取有误而 BM25 兜底原文正确时听谁的？"),
            ("• 下一阶段研究重点：", "建立失效标记（Tombstone）、时间戳权重与跨层真值裁决协议（Truth Arbitration），解决“都在记忆里却打架答错”的问题。"),
        ],
        bg=SOFT_RED, accent=CORAL, title_size=14, body_size=11.5,
    )

    OUT_PPTX.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT_PPTX))
    print(f"Saved native editable PPTX to {OUT_PPTX} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    build_pptx()
