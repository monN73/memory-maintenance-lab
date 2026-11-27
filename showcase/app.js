(function () {
  "use strict";

  const OV = window.__SHOWCASE_OVERVIEW__;
  if (!OV) {
    document.body.innerHTML = "<div style='padding:40px;font-family:sans-serif'>未找到 <code>data/overview.js</code>，请先运行 <code>python3 memory-maintenance-lab/scripts/build_showcase.py</code>。</div>";
    return;
  }

  window.__SHOWCASE_CASES__ = window.__SHOWCASE_CASES__ || {};

  const state = {
    view: "overview",
    strategyFamily: "all",
    matrixPlan: "plan_a",
    planKey: "plan_a",
    questionId: "89941a93",
    strategy: "N9",
    turnId: 185,
    loadingCaseKey: null,
    lastSyncedHash: "",
    lastHeaderKey: "",
    lastTimelineKey: "",
    lastPillsKey: "",
  };

  function esc(val) {
    if (val === null || val === undefined) return "";
    return String(val)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function pct(x) {
    if (x === null || x === undefined || Number.isNaN(Number(x))) return "—";
    return (Number(x) * 100).toFixed(1) + "%";
  }

  function num(x) {
    if (x === null || x === undefined) return "—";
    return Number(x).toLocaleString("en-US");
  }

  function syncHash(push) {
    try {
      const params = new URLSearchParams();
      params.set("view", state.view);
      if (state.view === "matrix") params.set("mplan", state.matrixPlan);
      if (state.view === "inspector") {
        params.set("plan", state.planKey);
        params.set("qid", state.questionId);
        params.set("strat", state.strategy);
        if (state.turnId != null) params.set("turn", String(state.turnId));
      }
      const nextHash = "#" + params.toString();
      state.lastSyncedHash = nextHash;
      if (window.location && window.location.hash !== nextHash) {
        if (push && window.history && typeof window.history.pushState === "function") {
          window.history.pushState(null, "", nextHash);
        } else if (window.history && typeof window.history.replaceState === "function") {
          window.history.replaceState(null, "", nextHash);
        }
      }
    } catch (_err) {
      // ignore sandboxed history errors
    }
  }

  function applyHashState() {
    try {
      const raw = (window.location && window.location.hash ? window.location.hash.slice(1) : "");
      if (!raw) return false;
      const params = new URLSearchParams(raw);
      const v = params.get("view");
      if (params.get("mplan") && OV.plans[params.get("mplan")]) {
        state.matrixPlan = params.get("mplan");
        document.querySelectorAll("[data-plan]").forEach((b) => {
          b.classList.toggle("active", b.dataset.plan === state.matrixPlan);
        });
        renderFailureMatrix();
      }
      if (params.get("plan") && OV.plans[params.get("plan")]) {
        state.planKey = params.get("plan");
        document.getElementById("selPlan").value = state.planKey;
      }
      populateQuestionSelect();
      if (params.get("qid") && OV.plans[state.planKey].questions[params.get("qid")]) {
        state.questionId = params.get("qid");
        document.getElementById("selQuestion").value = state.questionId;
      }
      populateStrategySelect();
      if (params.get("strat")) {
        state.strategy = params.get("strat");
        document.getElementById("selStrategy").value = state.strategy;
      }
      if (params.get("turn")) {
        const parsed = Number(params.get("turn"));
        state.turnId = Number.isFinite(parsed) ? parsed : null;
      }
      if (v && document.getElementById("view-" + v)) {
        switchView(v, false);
        return true;
      }
    } catch (_err) {
      return false;
    }
    return false;
  }

  function switchView(viewName, updateHash = true) {
    state.view = viewName;
    document.querySelectorAll(".nav-btn").forEach((b) => {
      b.classList.toggle("active", b.dataset.view === viewName);
    });
    document.querySelectorAll(".panel-view").forEach((sec) => {
      sec.classList.toggle("active", sec.id === "view-" + viewName);
    });
    if (updateHash) syncHash(true);
    if (viewName === "inspector") {
      ensureCaseAndRender();
    }
  }

  function openInspector(planKey, qid, strat, turnId) {
    state.planKey = planKey;
    state.questionId = qid;
    state.strategy = strat;
    state.turnId = turnId != null ? Number(turnId) : null;
    const mount = document.getElementById("turnDetailMount");
    if (mount && mount.style) mount.style.minHeight = "";
    document.getElementById("selPlan").value = planKey;
    populateQuestionSelect();
    document.getElementById("selQuestion").value = qid;
    populateStrategySelect();
    document.getElementById("selStrategy").value = strat;
    switchView("inspector", true);
    if (typeof window.scrollTo === "function") {
      window.scrollTo(0, 0);
    }
  }

  // ---------------- VIEW 1: OVERVIEW ----------------
  function renderOverview() {
    const sdev = OV.benchmarks.s_dev_30;
    const sku = OV.benchmarks.s_ku_10;
    const pa = OV.plans.plan_a.checklist;
    const pb = OV.plans.plan_b.checklist;

    document.getElementById("topbarMeta").textContent =
      `模型 ${OV.model} · 裁判 ${OV.judge_model} · 受控预算 B=${OV.budget.total_budget}（归档 ${OV.budget.archive_cap}） · 数据生成 ${OV.generated_at.slice(0, 19).replace("T", " ")}`;

    const kpis = [
      {
        label: "策略全集（N0–N14）",
        val: `${sdev.strategies_executed} / 15`,
        desc: "15 个策略全量执行（N0–N14，含 N6 分块 LLM 摘要）",
      },
      {
        label: "s-dev-30（30 题终态合并）",
        val: `${sdev.scored_cells} / ${sdev.total_cells}`,
        desc: `已结算且 scoring_pending=0；N1/N5 溢出 ${sdev.overflow_cells} 次、N4/N6 预算失败 ${sdev.error_cells} 次`,
      },
      {
        label: "Plan A 尾段快照探针（s-ku-10-v1）",
        val: `${num(pa.reusable_ok)} / ${num(pa.total)}`,
        desc: `${pa.termination_gaps} 个已解释终止缺口 · ${pa.rechecks_total} 复测（${pa.rechecks_instability} 例读出不稳定）· ${pa.cf_inbudget_ok} 跳过维护对照命中`,
      },
      {
        label: "Plan B 反转检查点（s-dev-30）",
        val: `${pb.reusable_ok} / ${pb.total}`,
        desc: `7 题 × N6, N8–N14 共同检查点 · ${pb.termination_gaps} 终止缺口 · ${pb.rechecks_consistent}/${pb.rechecks_total} 复测一致 · ${pb.cf_inbudget_ok} 跳过维护命中`,
      },
      {
        label: "oracle-5 高密度短历史基线",
        val: "5 题 × 15 策略",
        desc: `无噪声短历史下 N1/N6=100%、N4/N7/N8/N11/N12/N14=80%、N2/N3/N5/N9/N10/N13=60%`,
      },
    ];

    document.getElementById("overviewKpis").innerHTML = kpis
      .map(
        (k) => `
        <div class="kpi-card">
          <div class="label">${esc(k.label)}</div>
          <div class="val">${esc(k.val)}</div>
          <div class="desc">${esc(k.desc)}</div>
        </div>`
      )
      .join("");

    // Benchmark table
    const rowsHtml = OV.strategies
      .map((st) => {
        const d30 = st.s_dev_30;
        const k10 = st.s_ku_10;
        const o5 = st.oracle_5;
        if (st.status === "deferred") {
          return `
            <tr>
              <td><span class="badge badge-muted">${esc(st.id)}</span></td>
              <td><b>${esc(st.zh_name)}</b><div class="muted">${esc(st.name)}</div></td>
              <td colspan="7" class="muted">已实现分块再合并代码与单元测试；因辅助 LLM 压缩调用成本过高，v1.0 明确暂缓全量批跑。</td>
            </tr>`;
        }
        const fail30 = d30 ? (d30.overflow || 0) + (d30.errors || 0) : 0;
        const fail10 = k10 ? (k10.overflow || 0) + (k10.errors || 0) : 0;
        return `
          <tr>
            <td><span class="badge badge-accent">${esc(st.id)}</span></td>
            <td><b>${esc(st.zh_name)}</b><div class="muted">${esc(st.family_zh)} · ${esc(st.name)}</div></td>
            <td>
              ${
                d30
                  ? `<div class="bar-track"><div class="bar-fill" style="width:${(d30.end_to_end_success * 100).toFixed(1)}%"></div></div><b>${pct(d30.end_to_end_success)}</b> <span class="muted">(${d30.correct}/${d30.planned})</span>`
                  : "—"
              }
            </td>
            <td class="num">${d30 ? pct(d30.conditional_accuracy) + (fail30 ? ` <span class="badge badge-warn">失败${fail30}</span>` : "") : "—"}</td>
            <td class="num">${d30 && d30.avg_evidence_retention != null ? pct(d30.avg_evidence_retention) : "—"}</td>
            <td>
              ${
                k10
                  ? `<div class="bar-track"><div class="bar-fill ok" style="width:${(k10.end_to_end_success * 100).toFixed(1)}%"></div></div><b>${pct(k10.end_to_end_success)}</b> <span class="muted">(${k10.correct}/${k10.planned})</span>${fail10 ? ` <span class="badge badge-warn">失败${fail10}</span>` : ""}`
                  : "—"
              }
            </td>
            <td class="num">${k10 && k10.avg_evidence_retention != null ? pct(k10.avg_evidence_retention) : "—"}</td>
            <td class="num">${o5 ? `<b>${pct(o5.end_to_end_success)}</b> <span class="muted">(${o5.correct}/${o5.planned})</span>` : "—"}</td>
            <td class="num">${d30 ? num(d30.total_tokens) : "—"}</td>
            <td class="num">${k10 ? num(k10.total_tokens) : "—"}</td>
            <td class="num">${d30 ? `${Math.round(d30.total_latency_p50_ms)} ms` : "—"}</td>
          </tr>`;
      })
      .join("");

    document.getElementById("benchmarkTableWrap").innerHTML = `
      <table>
        <thead>
          <tr>
            <th>策略</th>
            <th>名称与定位</th>
            <th>s-dev-30 端到端正确率</th>
            <th class="num">s-dev-30 条件正确率</th>
            <th class="num">s-dev-30 平均证据留存</th>
            <th>s-ku-10 端到端正确率</th>
            <th class="num">s-ku-10 平均证据留存</th>
            <th class="num">oracle-5</th>
            <th class="num">s-dev-30 总 Token</th>
            <th class="num">s-ku-10 总 Token</th>
            <th class="num">s-dev-30 延迟 P50</th>
          </tr>
        </thead>
        <tbody>${rowsHtml}</tbody>
      </table>`;

    // Online probes denominator card
    const onlRows = ["N8", "N9", "N10"]
      .map((sid) => {
        const o = sdev.online_probes[sid];
        return `
          <tr>
            <td><span class="badge badge-accent">${sid}</span></td>
            <td class="num">${o.total_questions}</td>
            <td class="num">${o.questions_ge2_valid}</td>
            <td class="num">${o.questions_with_error}</td>
            <td class="num"><b>${o.questions_with_regression}</b></td>
            <td class="num">${o.scored} / ${o.total}</td>
          </tr>`;
      })
      .join("");
    document.getElementById("sdevOnlineSummary").innerHTML = `
      <table>
        <thead>
          <tr>
            <th>策略</th>
            <th class="num">总题数</th>
            <th class="num">≥2次有效观测</th>
            <th class="num">含错题数</th>
            <th class="num">观测到转错题数</th>
            <th class="num">有效评分 / 触发探针</th>
          </tr>
        </thead>
        <tbody>${onlRows}</tbody>
      </table>
      <div class="muted" style="margin-top:8px">
        注：N8 共触发 350 次在线探针，N9/N10 各触发 628 次；因此跨策略公平比较以 Plan B 共同离散检查点（483 格，每策略 69 格）为准。
      </div>`;

    // s-ku-10 evidence summary
    const etRows = sku.evidence_trajectories
      .map((et) => {
        const qmeta = OV.plans.plan_a.questions[et.question_id] || {};
        return `
          <tr>
            <td><button type="button" class="id-link-btn mono" data-jump-sku="${esc(et.question_id)}">${esc(et.question_id)}</button> ${qmeta.dev_exposed ? '<span class="badge badge-warn" title="该题曾在开发期用于检查抽取规则与快照重放验收">开发暴露</span>' : ""}</td>
            <td>${esc(et.question_type)}</td>
            <td class="num">${et.message_level_denominator}</td>
            <td class="num">${et.session_level_denominator}</td>
            <td class="num">${qmeta.first_turn ?? "?"}–${qmeta.last_turn ?? "?"} (${qmeta.turn_count ?? "?"}轮)</td>
          </tr>`;
      })
      .join("");
    document.getElementById("skuEvidenceSummary").innerHTML = `
      <table>
        <thead>
          <tr>
            <th>Question ID</th>
            <th>题型</th>
            <th class="num">消息级证据数</th>
            <th class="num">会话级证据数</th>
            <th class="num">尾段探测区间</th>
          </tr>
        </thead>
        <tbody>${etRows}</tbody>
      </table>`;

    document.querySelectorAll("[data-jump-sku]").forEach((a) => {
      a.addEventListener("click", (e) => {
        e.preventDefault();
        openInspector("plan_a", a.dataset.jumpSku, "N9", null);
      });
    });

    // s-dev-30 heatmap filter setup
    const types = Array.from(new Set(Object.values(sdev.questions).map((q) => q.question_type))).sort();
    const selType = document.getElementById("sdevTypeFilter");
    types.forEach((t) => {
      const opt = document.createElement("option");
      opt.value = t;
      opt.textContent = t;
      selType.appendChild(opt);
    });
    selType.addEventListener("change", renderSdevHeatmap);
    document.getElementById("sdevPlanBOnly").addEventListener("change", renderSdevHeatmap);
    renderSdevHeatmap();
    renderOracle5Heatmap();
  }

  function renderOutcomeCell(qid, s, c) {
    if (!c) return "<td>—</td>";
    let cls = "hm-fail";
    let sym = "×";
    if (c.outcome_category === "correct") {
      cls = "hm-correct";
      sym = "✓";
    } else if (c.outcome_category === "evidence_lost") {
      cls = "hm-lost";
      sym = "✗";
    } else if (c.outcome_category === "evidence_present_wrong_answer") {
      cls = "hm-present-wrong";
      sym = "!";
    } else if (c.outcome_category === "wrong_no_evidence_eval") {
      cls = "hm-no-ev";
      sym = "∅";
    }
    const title = `${qid} ${s}: ${c.outcome_category} (${c.status}, ctx=${c.final_ctx_tokens_est}, maint=${c.maintenance_events})`;
    return `<td style="text-align:center"><span class="hm-dot ${cls}" title="${esc(title)}">${sym}</span></td>`;
  }

  function renderSdevHeatmap() {
    const sdev = OV.benchmarks.s_dev_30;
    const typeFilter = document.getElementById("sdevTypeFilter").value;
    const planBOnly = document.getElementById("sdevPlanBOnly").checked;
    const strats = ["N0", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "N10", "N11", "N12", "N13", "N14"];
    const cellMap = new Map();
    sdev.results.forEach((r) => cellMap.set(`${r.question_id}::${r.strategy}`, r));

    const qids = Object.keys(sdev.questions).filter((qid) => {
      const q = sdev.questions[qid];
      if (typeFilter !== "all" && q.question_type !== typeFilter) return false;
      if (planBOnly && !q.in_plan_b) return false;
      return true;
    });

    const headCols = strats.map((s) => `<th style="text-align:center">${s}</th>`).join("");
    const bodyRows = qids
      .map((qid) => {
        const q = sdev.questions[qid];
        const tds = strats.map((s) => renderOutcomeCell(qid, s, cellMap.get(`${qid}::${s}`))).join("");
        const qLabel = q.in_plan_b
          ? `<button type="button" class="id-link-btn mono" data-open-planb="${esc(qid)}" title="点击进入 Plan B 共同检查点逐轮剖析">${esc(qid)}</button> <span class="badge badge-purple" title="纳入 s-dev-30 Plan B 共同检查点回放的 7 道反转样本之一">Plan B</span>`
          : `<span class="mono">${esc(qid)}</span>`;
        return `
          <tr>
            <td>${qLabel} ${q.is_abstention ? '<span class="badge badge-muted" title="LongMemEval 拒答题 (_abs)：标准答案为未提及，对遗忘不敏感">abs</span>' : ""}</td>
            <td>${esc(q.question_type)}</td>
            <td title="${esc(q.question)}">${esc(q.question.slice(0, 56))}${q.question.length > 56 ? "…" : ""}</td>
            ${tds}
          </tr>`;
      })
      .join("");

    document.getElementById("sdevHeatmapWrap").innerHTML = `
      <table>
        <thead><tr><th>题目</th><th>题型</th><th>问题摘要</th>${headCols}</tr></thead>
        <tbody>${bodyRows}</tbody>
      </table>`;

    document.querySelectorAll("[data-open-planb]").forEach((a) => {
      a.addEventListener("click", (e) => {
        e.preventDefault();
        openInspector("plan_b", a.dataset.openPlanb, "N9", null);
      });
    });
  }

  function renderOracle5Heatmap() {
    const mount = document.getElementById("oracle5HeatmapWrap");
    const o5 = OV.benchmarks.oracle_5;
    if (!mount || !o5 || !o5.questions || !o5.results) return;
    const strats = ["N0", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "N10", "N11", "N12", "N13", "N14"];
    const cellMap = new Map();
    o5.results.forEach((r) => cellMap.set(`${r.question_id}::${r.strategy}`, r));
    const headCols = strats.map((s) => `<th style="text-align:center">${s}</th>`).join("");
    const bodyRows = Object.keys(o5.questions)
      .map((qid) => {
        const q = o5.questions[qid];
        const tds = strats.map((s) => renderOutcomeCell(qid, s, cellMap.get(`${qid}::${s}`))).join("");
        return `
          <tr>
            <td><span class="mono">${esc(qid)}</span> ${q.is_abstention ? '<span class="badge badge-muted">abs</span>' : ""}</td>
            <td>${esc(q.question_type)}</td>
            <td title="${esc(q.question)}">${esc(q.question.slice(0, 56))}${q.question.length > 56 ? "…" : ""}</td>
            ${tds}
          </tr>`;
      })
      .join("");
    mount.innerHTML = `
      <table>
        <thead><tr><th>题目</th><th>题型</th><th>问题摘要</th>${headCols}</tr></thead>
        <tbody>${bodyRows}</tbody>
      </table>`;
  }

  // ---------------- VIEW 2: STRATEGIES ----------------
  function renderStrategies() {
    const fam = state.strategyFamily;
    const filtered = OV.strategies.filter((s) => fam === "all" || s.family === fam);
    const html = filtered
      .map((s) => {
        const d30 = s.s_dev_30;
        const k10 = s.s_ku_10;
        const o5 = s.oracle_5;
        return `
          <div class="strategy-card ${s.status === "deferred" ? "deferred" : ""}">
            <div class="top">
              <div>
                <div class="sid">${esc(s.id)} · ${esc(s.zh_name)}</div>
                <div class="muted">${esc(s.name)}</div>
              </div>
              <span class="badge ${s.status === "deferred" ? "badge-warn" : "badge-accent"}">${esc(s.family_zh)}</span>
            </div>
            <div class="kv">
              <div class="k">可见历史</div><div>${esc(s.visible_history)}</div>
              <div class="k">显式记忆</div><div>${esc(s.explicit_memory)}</div>
              <div class="k">预算契约</div><div>${esc(s.budget_rule)}</div>
              <div class="k">归档淘汰</div><div>${esc(s.archive_policy)}</div>
              <div class="k">维护机制</div><div>${esc(s.maintenance_mechanism)}</div>
              <div class="k">对照定位</div><div>${esc(s.control_note)}</div>
            </div>
            ${
              s.status === "deferred"
                ? `<div class="muted">当前作为 v1.0 明确暂缓项保留代码与单测，不计入 14 策略批量跑分表。</div>`
                : `<div class="strategy-metrics">
                    <div class="mcell">
                      <div class="muted">s-dev-30</div>
                      <b>${d30 ? pct(d30.end_to_end_success) : "—"}</b>
                      <div class="muted">${d30 ? `${d30.correct}/${d30.planned}` : ""}</div>
                      <div class="muted">${d30 ? `${num(d30.total_tokens)} tok` : ""}</div>
                    </div>
                    <div class="mcell">
                      <div class="muted">s-ku-10</div>
                      <b>${k10 ? pct(k10.end_to_end_success) : "—"}</b>
                      <div class="muted">${k10 ? `${k10.correct}/${k10.planned}` : ""}</div>
                      <div class="muted">${k10 ? `${num(k10.total_tokens)} tok` : ""}</div>
                    </div>
                    <div class="mcell">
                      <div class="muted">oracle-5</div>
                      <b>${o5 ? pct(o5.end_to_end_success) : "—"}</b>
                      <div class="muted">${o5 ? `${o5.correct}/${o5.planned}` : ""}</div>
                      <div class="muted">${o5 ? `${num(o5.total_tokens)} tok` : ""}</div>
                    </div>
                  </div>`
            }
          </div>`;
      })
      .join("");
    document.getElementById("strategyCardsGrid").innerHTML = html;
  }

  // ---------------- VIEW 3: FAILURE MATRIX ----------------
  function renderFeaturedCases() {
    document.getElementById("featuredCasesGrid").innerHTML = OV.featured_cases
      .map(
        (fc, idx) => `
        <div class="featured-card" data-fc-idx="${idx}">
          <div style="display:flex;justify-content:space-between;gap:8px;margin-bottom:6px">
            <span class="badge ${fc.tag.includes("跳过维护") ? "badge-ok" : fc.tag.includes("不稳定") ? "badge-warn" : "badge-accent"}">${esc(fc.tag)}</span>
            <span class="mono muted">${fc.plan_key === "plan_a" ? "Plan A" : "Plan B"}</span>
          </div>
          <div style="font-weight:700;font-size:13.5px;margin-bottom:4px">${esc(fc.title)}</div>
          <div class="muted" style="font-size:12px">${esc(fc.summary)}</div>
        </div>`
      )
      .join("");
    document.querySelectorAll(".featured-card").forEach((el) => {
      el.addEventListener("click", () => {
        const fc = OV.featured_cases[Number(el.dataset.fcIdx)];
        openInspector(fc.plan_key, fc.question_id, fc.strategy, fc.turn);
      });
    });
  }

  function renderFailureMatrix() {
    const planKey = state.matrixPlan;
    const plan = OV.plans[planKey];
    const cl = plan.checklist;
    const strats =
      planKey === "plan_a"
        ? ["N0", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "N10", "N11", "N12", "N13", "N14"]
        : ["N6", "N8", "N9", "N10", "N11", "N12", "N13", "N14"];

    document.getElementById("matrixTitle").textContent = `${plan.title} · 失效区间矩阵 (last_ok, first_reg]`;
    document.getElementById("planChecklistBanner").innerHTML =
      `核验结论：<span class="badge badge-ok">PASS</span> · 总格数 <b>${num(cl.total)}</b> · 有效完成 <b>${num(cl.reusable_ok)}</b> · 终止缺口 <b>${cl.termination_gaps}</b> · 意外缺失 <b>${cl.unexpected_missing}</b> · 口径失配 <b>${cl.mismatch_error}</b> · 边界复测 <b>${cl.rechecks_consistent}/${cl.rechecks_total}</b> 一致 · 跳过维护命中 <b>${cl.cf_inbudget_ok}/${cl.cf_total}</b>`;

    const rowMap = new Map();
    plan.rows.forEach((r) => rowMap.set(`${r.question_id}::${r.strategy}`, r));

    const headerCells = strats.map((s) => `<th style="text-align:center">${s}</th>`).join("");
    const bodyRows = Object.keys(plan.questions)
      .map((qid) => {
        const q = plan.questions[qid];
        const cells = strats
          .map((s) => {
            const r = rowMap.get(`${qid}::${s}`);
            if (!r) return "<td>—</td>";
            let cls = "cell-ok";
            let mainTxt = "未转错";
            let subTxt = `${r.valid_observations}轮全对`;
            let jumpTurn = q.first_turn || (q.turns && q.turns[0]);

            if (r.run_status && r.run_status !== "completed" && !r.regression_interval && r.valid_observations === 0) {
              cls = "cell-fail";
              mainTxt = r.run_status === "overflow" ? "溢出" : "预算失败";
              subTxt = `缺口${r.termination_gaps}`;
            } else if (r.regression_interval) {
              jumpTurn = r.first_regression_turn;
              mainTxt = r.regression_interval;
              if (r.has_inbudget_cf_win) {
                cls = "cell-cf-win";
                subTxt = "★跳过答对→维护错";
              } else if (r.causal_framing && r.causal_framing.includes("发生维护")) {
                cls = "cell-maint-reg";
                subTxt = `含维护事件(${(r.interval_events || []).length})`;
              } else {
                cls = "cell-drift-reg";
                subTxt = "滑出/读出波动";
              }
            } else if (r.first_observation_incorrect) {
              cls = "cell-init-err";
              jumpTurn = r.first_observed_error_turn;
              mainTxt = `T${r.first_observed_error_turn}首错`;
              subTxt = "首次观测即错";
            } else if (r.run_status && r.run_status !== "completed") {
              subTxt = `${r.valid_observations}轮 (${r.run_status})`;
            }

            if (r.instability_turns && r.instability_turns.length) {
              subTxt += ` ⚠不稳定`;
            }

            const cellTip = `${qid} · ${s}：${mainTxt}（${subTxt}）。点击直达对应轮次三线证据`;
            return `<td><div class="matrix-cell ${cls}" title="${esc(cellTip)}" data-m-plan="${planKey}" data-m-qid="${esc(qid)}" data-m-strat="${s}" data-m-turn="${jumpTurn ?? ""}">${esc(mainTxt)}<div style="font-size:10px;opacity:0.85">${esc(subTxt)}</div></div></td>`;
          })
          .join("");

        return `
          <tr>
            <td>
              <div class="mono"><b>${esc(qid)}</b></div>
              <div class="muted" style="font-size:11px">${esc(q.question_type)}${q.dev_exposed ? ' · <span class="badge badge-warn" title="开发阶段曾用于规则与重放调试">开发暴露</span>' : ""}${q.is_abstention ? ' · <span class="badge badge-muted" title="LongMemEval 拒答题 (_abs)：标准答案为未提及，对遗忘不敏感">abs</span>' : ""}</div>
            </td>
            <td style="min-width:190px">${esc(q.question.slice(0, 68))}${q.question.length > 68 ? "…" : ""}<div class="muted" style="font-size:11px">期望：${esc(String(q.expected).slice(0, 48))}</div></td>
            ${cells}
          </tr>`;
      })
      .join("");

    document.getElementById("regressionMatrixWrap").innerHTML = `
      <table>
        <thead><tr><th>题目 ID</th><th>问题与标准答案</th>${headerCells}</tr></thead>
        <tbody>${bodyRows}</tbody>
      </table>`;

    document.querySelectorAll(".matrix-cell").forEach((el) => {
      el.addEventListener("click", () => {
        openInspector(el.dataset.mPlan, el.dataset.mQid, el.dataset.mStrat, el.dataset.mTurn || null);
      });
    });
  }

  // ---------------- VIEW 4: TURN-BY-TURN INSPECTOR ----------------
  function populateQuestionSelect() {
    const plan = OV.plans[state.planKey];
    const sel = document.getElementById("selQuestion");
    sel.innerHTML = Object.keys(plan.questions)
      .map((qid) => {
        const q = plan.questions[qid];
        return `<option value="${esc(qid)}">${esc(qid)} · ${esc(q.question_type)} · ${esc(q.question.slice(0, 40))}</option>`;
      })
      .join("");
    if (!plan.questions[state.questionId]) {
      state.questionId = Object.keys(plan.questions)[0];
    }
    sel.value = state.questionId;
  }

  function populateStrategySelect() {
    const strats =
      state.planKey === "plan_a"
        ? ["N0", "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "N10", "N11", "N12", "N13", "N14"]
        : ["N6", "N8", "N9", "N10", "N11", "N12", "N13", "N14"];
    const sel = document.getElementById("selStrategy");
    sel.innerHTML = strats.map((s) => `<option value="${s}">${s}</option>`).join("");
    if (!strats.includes(state.strategy)) {
      state.strategy = strats[0];
    }
    sel.value = state.strategy;
  }

  function ensureCaseAndRender() {
    const key = `${state.planKey}__${state.questionId}`;
    const mount = document.getElementById("turnDetailMount");
    if (window.__SHOWCASE_CASES__[key]) {
      renderInspector(window.__SHOWCASE_CASES__[key]);
      return;
    }
    if (state.loadingCaseKey === key) return;
    state.loadingCaseKey = key;
    if (mount && mount.style && mount.offsetHeight) {
      mount.style.minHeight = Math.max(mount.offsetHeight, 480) + "px";
    }
    mount.innerHTML = `<div class="card muted">正在加载 <code>${esc(key)}.js</code> 轨迹与维护快照证据…</div>`;

    const script = document.createElement("script");
    script.src = `data/cases/${key}.js`;
    script.onload = () => {
      state.loadingCaseKey = null;
      if (window.__SHOWCASE_CASES__[key]) {
        renderInspector(window.__SHOWCASE_CASES__[key]);
      } else {
        mount.innerHTML = `<div class="card">加载失败：未找到数据包 <code>${esc(key)}</code>。</div>`;
      }
    };
    script.onerror = () => {
      state.loadingCaseKey = null;
      mount.innerHTML = `<div class="card">无法读取 <code>data/cases/${esc(key)}.js</code>。</div>`;
    };
    document.body.appendChild(script);
  }

  function _snapshotReadoutHtml(sObj, currRec) {
    return `
      <div style="display:flex;justify-content:space-between;margin-bottom:4px">
        <b>当前轮快照探针读出（snapshot_readout · T${state.turnId}）</b>
        ${
          currRec.status === "ok"
            ? currRec.correct
              ? '<span class="badge badge-ok">探针 judge: yes</span>'
              : '<span class="badge badge-err">探针 judge: no</span>'
            : `<span class="badge badge-muted">${esc(currRec.status)}</span>`
        }
      </div>
      <div class="muted" style="font-size:12px;margin-bottom:4px">当前查看策略 ${esc(state.strategy)} · 失效区间：<b>${esc(sObj.audit.regression_interval || "无退化")}</b> · 首错轮：${sObj.audit.first_observed_error_turn ? "T" + sObj.audit.first_observed_error_turn : "—"}</div>
      <div>${esc(currRec.status === "ok" ? currRec.hypothesis : `（${currRec.status}：${currRec.termination_reason || sObj.termination_reason || "运行已提前终止"}）`)}</div>`;
  }

  function renderInspector(caseData) {
    const mountEl = document.getElementById("turnDetailMount");
    if (mountEl && mountEl.style && mountEl.offsetHeight) {
      mountEl.style.minHeight = Math.max(mountEl.offsetHeight, 480) + "px";
    }

    const sObj = caseData.strategies[state.strategy] || Object.values(caseData.strategies)[0];
    if (!sObj) return;
    state.strategy = sObj.strategy;
    document.getElementById("selStrategy").value = state.strategy;

    const probedTurns = caseData.probed_turns;
    if (state.turnId == null || !probedTurns.includes(state.turnId)) {
      state.turnId = sObj.audit.first_regression_turn || sObj.audit.first_observed_error_turn || probedTurns[0];
    }
    syncHash(false);

    const df = sObj.direct_final || {};
    const currRec = sObj.turns[String(state.turnId)] || {};
    const currEx = caseData.exchanges[String(state.turnId)];
    const headerKey = `${caseData.plan_key}::${caseData.question_id}::${state.strategy}`;
    const snapBox = document.getElementById("inspectorSnapshotBox");

    if (state.lastHeaderKey === headerKey && snapBox) {
      snapBox.innerHTML = _snapshotReadoutHtml(sObj, currRec);
    } else {
      state.lastHeaderKey = headerKey;
      const badges = [
        `<span class="badge badge-accent">${esc(caseData.question_type)}</span>`,
        `<span class="badge badge-muted">总轮数 ${caseData.total_turns}</span>`,
        caseData.dev_exposed ? `<span class="badge badge-warn" title="该题曾在开发期用于检查抽取规则与快照重放验收">开发暴露样本</span>` : "",
        caseData.is_abstention ? `<span class="badge badge-purple" title="LongMemEval 拒答题 (_abs)：标准答案为未提及，遗忘历史后仍会拒答">拒答样本 (_abs)</span>` : "",
        sObj.audit.instability_turns && sObj.audit.instability_turns.length
          ? `<span class="badge badge-warn" title="同一冻结请求 --recheck 两次回答实质翻转">⚠ 检出 readout_instability: T${sObj.audit.instability_turns.join(", T")}</span>`
          : "",
      ].join(" ");

      document.getElementById("questionHeaderCard").innerHTML = `
        <div class="card-head">
          <div>
            <h2 style="margin-bottom:4px">${esc(caseData.question_id)} · ${esc(caseData.question)}</h2>
            <div style="display:flex;flex-wrap:wrap;gap:6px">${badges}</div>
          </div>
          <div style="text-align:right">
            <div class="muted">标准参考答案</div>
            <div><b>${esc(caseData.expected)}</b></div>
            ${caseData.evidence_turn_ids.length ? `<div class="muted">证据轮次：T${caseData.evidence_turn_ids.join(", T")}</div>` : ""}
          </div>
        </div>
        ${caseData.validity_note ? `<div class="muted" style="margin-bottom:10px">有效性边界说明：${esc(caseData.validity_note)}</div>` : ""}
        <div class="two-col">
          <div class="quote-box header-quote-fixed">
            <div style="display:flex;justify-content:space-between;margin-bottom:4px">
              <b>原批次最终回答（direct_final · turn_${caseData.total_turns + 1}）</b>
              ${
                df.judge_label === true
                  ? '<span class="badge badge-ok">终态 judge: yes</span>'
                  : df.judge_label === false
                  ? '<span class="badge badge-err">终态 judge: no</span>'
                  : `<span class="badge badge-muted">${esc(df.status)}</span>`
              }
            </div>
            <div class="muted" style="font-size:12px;margin-bottom:4px">归因类别：${esc(df.outcome_category)} · 终态上下文 ${num(df.final_ctx_tokens_est)} tokens · 维护事件 ${num(df.maintenance_events)} 次</div>
            <div>${esc(df.hypothesis || `（无输出，终止原因：${sObj.termination_reason || df.status}）`)}</div>
          </div>
          <div class="quote-box header-quote-fixed" id="inspectorSnapshotBox">
            ${_snapshotReadoutHtml(sObj, currRec)}
          </div>
        </div>`;
    }

    // Multi-strategy timeline strip: rebuild only when switching question
    const timelineKey = `${caseData.plan_key}::${caseData.question_id}`;
    const tlContainer = document.getElementById("multiStrategyTimeline");
    if (state.lastTimelineKey !== timelineKey) {
      state.lastTimelineKey = timelineKey;
      const tlHtml = Object.keys(caseData.strategies)
        .map((sid) => {
          const st = caseData.strategies[sid];
          const ticks = probedTurns
            .map((t) => {
              const tr = st.turns[String(t)];
              let cls = "gap";
              if (tr && tr.status === "ok") cls = tr.correct ? "ok" : "err";
              if (st.audit.first_regression_turn === t) cls += " reg";
              if (sid === state.strategy && t === state.turnId) cls += " curr";
              return `<div class="tl-tick ${cls}" data-tl-strat="${sid}" data-tl-turn="${t}" title="${sid} T${t}: ${tr ? (tr.status === "ok" ? (tr.correct ? "yes" : "no") : tr.status) : "missing"}"></div>`;
            })
            .join("");
          const regLbl = st.audit.regression_interval
            ? st.audit.regression_interval
            : st.audit.first_observation_incorrect
            ? `T${st.audit.first_observed_error_turn}首错`
            : st.run_status !== "completed"
            ? st.run_status
            : "未转错";
          return `
            <div class="tl-row" data-tl-row="${sid}">
              <div><span class="badge ${sid === state.strategy ? "badge-accent" : "badge-muted"}" data-tl-badge="${sid}">${sid}</span></div>
              <div class="muted mono" style="font-size:11px">${esc(regLbl)}</div>
              <div class="tl-strip">${ticks}</div>
            </div>`;
        })
        .join("");
      tlContainer.innerHTML = tlHtml;
      tlContainer.querySelectorAll(".tl-tick").forEach((el) => {
        el.addEventListener("click", (e) => {
          e.preventDefault();
          e.stopPropagation();
          state.strategy = el.dataset.tlStrat;
          state.turnId = Number(el.dataset.tlTurn);
          document.getElementById("selStrategy").value = state.strategy;
          renderInspector(caseData);
        });
      });
    } else {
      tlContainer.querySelectorAll("[data-tl-badge]").forEach((b) => {
        const isActive = b.dataset.tlBadge === state.strategy;
        b.classList.toggle("badge-accent", isActive);
        b.classList.toggle("badge-muted", !isActive);
      });
      tlContainer.querySelectorAll(".tl-tick.curr").forEach((el) => el.classList.remove("curr"));
      const activeTick = tlContainer.querySelector(`.tl-tick[data-tl-strat="${state.strategy}"][data-tl-turn="${state.turnId}"]`);
      if (activeTick) activeTick.classList.add("curr");
    }

    // Turn pills: rebuild only when switching question or strategy
    const pillsEl = document.getElementById("turnPills");
    const pillsKey = `${caseData.plan_key}::${caseData.question_id}::${state.strategy}`;
    if (state.lastPillsKey !== pillsKey) {
      state.lastPillsKey = pillsKey;
      document.getElementById("currentStrategyStats").textContent =
        `有效观测 ${sObj.audit.valid_observations ?? "—"} · 终止缺口 ${sObj.audit.termination_gaps ?? 0} · 定性：${sObj.audit.causal_framing || "—"}`;
      pillsEl.innerHTML = probedTurns
        .map((t) => {
          const tr = sObj.turns[String(t)];
          let cls = "gap";
          if (tr && tr.status === "ok") cls = tr.correct ? "ok" : "err";
          if (sObj.audit.first_regression_turn === t) cls += " reg-point";
          if (t === state.turnId) cls += " active";
          const dots = [
            tr && tr.events && tr.events.length ? '<span class="dot-maint" title="含维护事件"></span>' : "",
            tr && tr.counterfactual ? '<span class="dot-cf" title="含跳过维护对照"></span>' : "",
            tr && tr.rechecks && tr.rechecks.length ? '<span class="dot-rc" title="含边界复测"></span>' : "",
          ].join("");
          return `<button type="button" class="turn-pill ${cls}" data-pick-turn="${t}">T${t}${dots}</button>`;
        })
        .join("");
      pillsEl.querySelectorAll("[data-pick-turn]").forEach((btn) => {
        btn.addEventListener("click", (e) => {
          e.preventDefault();
          e.stopPropagation();
          state.turnId = Number(btn.dataset.pickTurn);
          renderInspector(caseData);
        });
      });
    } else {
      pillsEl.querySelectorAll(".turn-pill.active").forEach((b) => b.classList.remove("active"));
      const activePill = pillsEl.querySelector(`[data-pick-turn="${state.turnId}"]`);
      if (activePill) activePill.classList.add("active");
    }

    // Detail Mount
    renderTurnDetail(caseData, sObj, currRec, currEx);
  }

  function renderTurnDetail(caseData, sObj, rec, ex) {
    const mount = document.getElementById("turnDetailMount");
    if (!rec || rec.status !== "ok") {
      mount.innerHTML = `
        <div class="card">
          <h3>T${state.turnId} · 观测缺口（${esc((rec && rec.status) || "gap")}）</h3>
          <div class="muted">原运行在该轮已进入终止状态（<code>${esc((rec && rec.termination_reason) || sObj.termination_reason || "overflow / error")}</code>），无完整有效快照可重放。请切换至终止前轮次或其他受控策略。</div>
        </div>`;
      return;
    }

    const sd = rec.snapshot_diff;
    const row1Html = sd
      ? `
        <div class="evidence-row">
          <h4>第 1 行 · 相邻快照状态变化（<code>T${state.turnId - 1} → T${state.turnId}</code>）</h4>
          <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px">
            <span class="badge badge-accent">近期窗口保留：${sd.curr_kept_ids.length} 轮 (${sd.curr_kept_ids.length ? `T${sd.curr_kept_ids[0]}..T${sd.curr_kept_ids[sd.curr_kept_ids.length - 1]}` : "空"})</span>
            <span class="badge badge-ok">+新入窗口：${sd.added_kept_ids.length ? "T" + sd.added_kept_ids.join(", T") : "无"}</span>
            <span class="badge badge-err">-移出窗口：${sd.dropped_kept_ids.length ? "T" + sd.dropped_kept_ids.join(", T") : "无"}</span>
            <span class="badge badge-purple">归档池：${sd.prev_archived_count} → ${sd.curr_archived_count} 轮</span>
            ${sd.dropped_archive_ids.length ? `<span class="badge badge-err">-归档永久淘汰：T${sd.dropped_archive_ids.join(", T")}</span>` : ""}
            <span class="badge badge-muted">显式记忆长度：${num(sd.prev_mem_len)} → ${num(sd.curr_mem_len)} 字符</span>
          </div>
          ${sd.curr_mem_preview ? `<details><summary>展开查看轮末显式记忆 snapshot.memory_text 预览</summary><pre class="code-box">${esc(sd.curr_mem_preview)}</pre></details>` : '<div class="muted">本策略在快照中无独立 memory_text 正文（如 N0/N1/N2/N7/N8/N11/N12）。</div>'}
        </div>`
      : `<div class="evidence-row"><h4>第 1 行 · 相邻快照状态变化</h4><div class="muted">暂无前置快照对比数据。</div></div>`;

    const row2Html =
      rec.events && rec.events.length
        ? `
        <div class="evidence-row">
          <h4>第 2 行 · 当轮维护算子事件（<code>events.jsonl</code>） <span class="badge badge-purple">共 ${rec.events.length} 条</span></h4>
          ${rec.events
            .map(
              (ev) => `
              <div style="margin-bottom:10px">
                <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:6px">
                  <span class="badge badge-purple">${esc(ev.mechanism)}</span>
                  <span class="badge badge-muted">phase: ${esc(ev.phase)}</span>
                  <span class="badge badge-err">窗口移出：${ev.evicted_turn_ids.length ? "T" + ev.evicted_turn_ids.join(", T") : "无"}</span>
                  ${ev.archive_evicted_turn_ids.length ? `<span class="badge badge-err">归档淘汰：T${ev.archive_evicted_turn_ids.join(", T")}</span>` : ""}
                  <span class="badge badge-muted">token: ${num(ev.input_tokens)} → ${num(ev.output_tokens)}</span>
                </div>
                <div class="two-col">
                  <div>
                    <div class="muted"><b>before_text</b>（${esc(ev.before_label)}）</div>
                    <pre class="code-box">${esc(ev.before_text || "（空）")}</pre>
                  </div>
                  <div>
                    <div class="muted"><b>after_text</b>（${esc(ev.after_label)}）</div>
                    <pre class="code-box">${esc(ev.after_text || "（空）")}</pre>
                  </div>
                </div>
              </div>`
            )
            .join("")}
        </div>`
        : `<div class="evidence-row"><h4>第 2 行 · 当轮维护算子事件（<code>events.jsonl</code>）</h4><div class="muted">本轮（T${state.turnId}）未触发维护事件（候选长度未越过 trigger=6553）。</div></div>`;

    const row3Html = `
      <div class="evidence-row">
        <h4>第 3 行 · 冻结快照探针读出与实际注入上下文（<code>records/</code>）
          ${rec.correct ? '<span class="badge badge-ok">yes (答对)</span>' : '<span class="badge badge-err">no (答错)</span>'}
        </h4>
        <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px">
          <span class="badge badge-accent">检索召回轮次：${rec.retrieved_turn_ids.length ? "T" + rec.retrieved_turn_ids.join(", T") : "无"}</span>
          <span class="badge badge-muted">归档白名单：${rec.archived_turn_ids.length} 轮</span>
          <span class="badge badge-muted">输入 tokens：${num(rec.probe_tokens_in)}</span>
          <span class="badge badge-muted mono">sha256: ${esc((rec.request_sha256 || "").slice(0, 12))}</span>
        </div>
        <div class="quote-box"><b>模型探针回答：</b>${esc(rec.hypothesis)}</div>
        <details>
          <summary>展开查看注入给回答模型的完整记忆正文（memory_text_injected）</summary>
          <pre class="code-box">${esc(rec.memory_text_injected || "（无外部记忆注入）")}</pre>
        </details>
      </div>`;

    // Counterfactual + Recheck panel
    let extraPanels = "";
    if (rec.counterfactual || (rec.rechecks && rec.rechecks.length)) {
      const cf = rec.counterfactual;
      const rcs = rec.rechecks || [];
      extraPanels = `
        <div class="card">
          <div class="card-head">
            <h3>边界验证：同轮跳过维护对照（<code>--skip-maintenance</code>）与边界复测（<code>--recheck</code>）</h3>
          </div>
          <div class="cf-grid">
            ${
              cf
                ? cf.fits_budget
                  ? `<div class="cf-box ${cf.correct ? "win" : "lose"}">
                      <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                        <b>本轮跳过维护的对照读出（<code>turn_${state.turnId - 1} + ex_${state.turnId}</code>）</b>
                        <span class="badge ${cf.correct ? "badge-ok" : "badge-err"}">${cf.correct ? "yes (跳过维护答对)" : "no"}</span>
                      </div>
                      <div class="muted" style="font-size:12px;margin-bottom:6px">
                        估算 ${num(cf.estimated_tokens)} ≤ ${num(cf.budget_cap_tokens)}（fits_budget=true） · 召回：${cf.retrieved_turn_ids.length ? "T" + cf.retrieved_turn_ids.join(", T") : "无"} · sha256: <code>${esc((cf.request_sha256 || "").slice(0, 10))}</code>
                      </div>
                      <div class="quote-box">${esc(cf.hypothesis)}</div>
                      ${cf.correct && !rec.correct ? '<div style="margin-top:8px;font-weight:700;color:#065f46;font-size:12.5px">✓ 结论：同轮配对重放中，跳过维护时答对，按原维护结果读出时答错。</div>' : ""}
                      <details><summary>查看对照读出注入上下文</summary><pre class="code-box">${esc(cf.memory_text_injected || "（无）")}</pre></details>
                    </div>`
                  : `<div class="cf-box">
                      <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                        <b>本轮跳过维护的对照读出（预算检查）</b>
                        <span class="badge badge-warn">over_budget (${num(cf.estimated_tokens)} &gt; ${num(cf.budget_cap_tokens)})</span>
                      </div>
                      <div class="muted">该策略在 <code>T${state.turnId - 1}</code> 已接近 8192 上限，跳过维护直接叠加本轮对话会超出统一预算上限，因此按协议不构造超限反事实回答，仅展示相邻快照与维护事件差异。</div>
                    </div>`
                : ""
            }
            ${rcs
              .map(
                (rc) => `
                <div class="cf-box ${rc.classification === "consistent" ? "" : "lose"}">
                  <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                    <b>同一冻结输入独立复测（<code>--recheck</code>）</b>
                    <span class="badge ${rc.classification === "consistent" ? "badge-ok" : "badge-warn"}">${esc(rc.classification)}</span>
                  </div>
                  <div class="muted" style="font-size:12px;margin-bottom:6px">
                    首次评分：<b>${rc.primary_correct}</b> · 复测评分：<b>${rc.correct}</b>（judge_raw: ${esc(rc.judge_raw)}）
                  </div>
                  <div class="quote-box">${esc(rc.hypothesis)}</div>
                </div>`
              )
              .join("")}
          </div>
        </div>`;
    }

    // Raw conversation exchange viewer (dataset original user & assistant)
    const convHtml = ex
      ? `
        <div class="card">
          <div class="card-head">
            <div>
              <h3>原始数据集对话上下文（T${state.turnId} · session <code>${esc(ex.session_id)}</code> · ${esc(ex.timestamp)}）</h3>
              <div class="muted">此处展示数据集中的原始用户发言与原始助手回复（<code>dataset assistant original</code>），请勿与右侧探针回答或终态回答混淆。</div>
            </div>
            ${ex.has_answer ? '<span class="badge badge-warn">原数据集标注证据轮 (has_answer)</span>' : ""}
          </div>
          <div class="two-col">
            <div class="quote-box"><b>User (T${state.turnId})：</b>\n${esc(ex.user_text)}</div>
            <div class="quote-box"><b>Dataset Assistant Original (T${state.turnId})：</b>\n${esc(ex.assistant_text || "（无助手文本）")}</div>
          </div>
        </div>`
      : "";

    mount.innerHTML = `
      <div class="card">
        <div class="card-head">
          <h3>三线证据面板 · T${state.turnId}（${esc(state.strategy)}）</h3>
          <div class="muted">来源批次：<code>${esc(rec.source_batch || sObj.source_batch)}</code></div>
        </div>
        <div class="three-rows">${row1Html}${row2Html}${row3Html}</div>
      </div>
      ${extraPanels}
      ${convHtml}`;
  }

  // ---------------- EVENTS ----------------
  document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.addEventListener("click", () => switchView(btn.dataset.view));
  });

  const btnGlossary = document.getElementById("btnOpenGlossary");
  if (btnGlossary) {
    btnGlossary.addEventListener("click", () => switchView("glossary"));
  }

  document.querySelectorAll("#strategyFamilyFilter .pill-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#strategyFamilyFilter .pill-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.strategyFamily = btn.dataset.family;
      renderStrategies();
    });
  });

  document.querySelectorAll("[data-plan]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("[data-plan]").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.matrixPlan = btn.dataset.plan;
      renderFailureMatrix();
    });
  });

  document.getElementById("selPlan").addEventListener("change", (e) => {
    state.planKey = e.target.value;
    state.turnId = null;
    populateQuestionSelect();
    populateStrategySelect();
    ensureCaseAndRender();
  });

  document.getElementById("selQuestion").addEventListener("change", (e) => {
    state.questionId = e.target.value;
    state.turnId = null;
    ensureCaseAndRender();
  });

  document.getElementById("selStrategy").addEventListener("change", (e) => {
    state.strategy = e.target.value;
    state.turnId = null;
    ensureCaseAndRender();
  });

  document.getElementById("btnPrevTurn").addEventListener("click", () => {
    const c = window.__SHOWCASE_CASES__[`${state.planKey}__${state.questionId}`];
    if (!c) return;
    const idx = c.probed_turns.indexOf(state.turnId);
    if (idx > 0) {
      state.turnId = c.probed_turns[idx - 1];
      renderInspector(c);
    }
  });

  document.getElementById("btnNextTurn").addEventListener("click", () => {
    const c = window.__SHOWCASE_CASES__[`${state.planKey}__${state.questionId}`];
    if (!c) return;
    const idx = c.probed_turns.indexOf(state.turnId);
    if (idx >= 0 && idx < c.probed_turns.length - 1) {
      state.turnId = c.probed_turns[idx + 1];
      renderInspector(c);
    }
  });

  document.getElementById("btnJumpReg").addEventListener("click", () => {
    const c = window.__SHOWCASE_CASES__[`${state.planKey}__${state.questionId}`];
    if (!c) return;
    const sObj = c.strategies[state.strategy];
    if (!sObj) return;
    const target = sObj.audit.first_regression_turn || sObj.audit.first_observed_error_turn;
    if (target != null) {
      state.turnId = target;
      renderInspector(c);
    }
  });

  if (typeof window.addEventListener === "function") {
    window.addEventListener("hashchange", () => {
      if (window.location && window.location.hash === state.lastSyncedHash) return;
      applyHashState();
    });
  }

  // Init
  renderOverview();
  renderStrategies();
  renderFeaturedCases();
  renderFailureMatrix();
  populateQuestionSelect();
  populateStrategySelect();
  applyHashState();
})();
