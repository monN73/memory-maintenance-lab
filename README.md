# Memory Maintenance Lab

目标：围绕 **“不同的外部记忆维护方法会造成什么影响”** 建立可复现、可归因的参考 Baseline 体系，在严格控制上下文预算与历史回放的条件下，系统观察信息在**写入、更新、撤回、压缩、淘汰与按需读取**中如何保留或失效。

**状态：direct runner 已实现 N0–N14。P3 验证策略写入、快照重放和事件溯源。P4 在冻结的 LongMemEval s-ku-10 采集 140 条计划轨迹（跳过 N6），交付成绩与成本表及逐证据维护轨迹。**

## docs

1. [research-protocol.md](docs/research-protocol.md) 记录外部记忆维护的策略、预算和对照。
2. [evaluation.md](docs/evaluation.md)：数据集体系（全面采用 LongMemEval 公开基准：`oracle-5`、`s-dev-30`、`s-ku-10`，后续仅扩展其他公开基准，不再新建自建集）、逐事实三类状态迁移口径（漏写 / 遗忘 / 正确删除旧值）、确定性/语义维护审计指标与结果展示。
3. [agent-integration.md](docs/agent-integration.md)：策略可组合维度拆解，以及**暂缓执行**的 Agent 原生记忆/外部编排交互设计储备。
4. [roadmap.md](docs/roadmap.md) 记录小样本验收、统一跑批和后续扩展。

文档引用统一使用完整英文文件名，正文继续使用中文说明。每份文档按主题分节，后续内容优先加入已有主题。

## 项目文件

- `configs/`：pilot、策略注册、编排注册、实验单元清单；均为 design_only/proposed，不是 Hermes 原生配置。
- `templates/`：run、turn、checkpoint、维护事件、能力审计、场景和报告模板；不是完整 JSON Schema。
- `data/scenarios/` 与 `data/annotations/`：输入和评估标注分离。
- `prompts/`：维护提示词；`examples/`：明确标记的人工展示数据。
- `runs/` 与 `reports/`：端到端运行及评分；`diagnostics/`：零 API 维护回放；`scripts/`：数据准备、静态设计校验和离线回放。

各方法的窗口、触发阈值和预算分配见 [research-protocol.md](docs/research-protocol.md)。`N1/N5` 使用单独指定的模型总上下文容量（`131072`），`N2–N4/N6/N7–N14` 默认采用 `8K` 受控回答输入预算（其中 `M ≤ 2048`、近期窗口维护时 `≤ 4096`）；不同预算或归档容量下的结果需分别解释。

研究聚焦不同方法怎样改变外部记忆。P3 使用 LongMemEval 的 01493427 和 45dc21b6 两题作开发验收，P4 在同一冻结清单上报告任务结果和逐证据轨迹。Hermes 接入属于后续宿主验证。

静态校验：在本目录执行 `.venv/bin/python scripts/validate_design.py`。它只检查 JSON、文件引用、注册与实验数量一致性，不验证模型行为、原生支持、完整 schema 或科学有效性。

离线归档维护回放示例（在项目目录中执行，使用已完成的 N7 批次）：

```bash
.venv/bin/python scripts/replay_archive_maintenance.py \
  runs/ecnu-plus/longmemeval-s-dev-30-n5n7--ecnu--20261118T151945788798+0800 \
  --archive-tokens 8192 --output diagnostics/archive-maintenance/s-dev-30
```

## 选择 API 服务

默认使用 DeepSeek；指定 `--provider ecnu` 时，从环境变量或 `~/.hermes/.env` 读取 `ECNU_API_KEY` 和 `ECNU_BASE_URL`，默认模型为 `ecnu-plus`。环境变量优先，Base URL 原样拼接 `/chat/completions`，请在配置中包含服务要求的版本路径。不要在命令行传入密钥。

在项目目录中，用独立批次运行 ECNU：

```bash
cd /home/monday73/memManagement/memory-maintenance-lab
python3 -m venv .venv  # 首次创建
source .venv/bin/activate
python src/runner.py --mode longmemeval --provider ecnu --strategies N0,N2,N3,N4 \
  --dataset data/longmemeval/longmemeval_oracle.json \
  --manifest data/manifests/oracle-5.json --probe-schedule final_only --workers 4

# 默认自动选取最新批次；如需指定历史批次，可追加 --batch-id <批次目录名>
python src/eval_and_report.py --provider ecnu --generation-provider ecnu --generation-model ecnu-plus \
  --dataset data/longmemeval/longmemeval_oracle.json --workers 4
```

> 说明：
> - `runner.py` 与 `eval_and_report.py` 的 `--batch-id` 均已改为可选：前者不传时自动生成 `<mode>-<manifest/dataset>--<服务>--<+0800时间戳>`，后者不传时自动选取最新完成的批次。
> - `--workers` 控制并发；应按实际服务限流配置。如只需在触发压缩/淘汰时探测或只测最终问答，可将 `--probe-schedule` 改为 `on_maintenance` 或 `final_only`。

可以用 `--model` 改变回答和压缩模型，用 `--judge-model` 改变评分模型；评分时的 `--provider` 选择评分服务，与生成时的服务可独立设置。使用不同 judge 的得分须分别说明。ECNU 使用 Qwen 的信息按用户提供记录；当前未验证服务端具体模型版本。

每次运行自动创建 `runs/<模型>/<batch-id>--<生成服务>--<东八区时间戳>/`，预测文件名为 `predictions_<策略>.jsonl`。报告与评分结果直接放在 `reports/<生成模型>/<批次>/`，文件名为 `report.html`、`summary.json`、`eval_N4.jsonl` 等。同一次实验在 runs 与 reports 下使用相同的批次目录名。生成和评分服务仍记录在文件内容中。评分时可使用 runner 输出的完整批次目录名；如果同名批次只有一次，也可直接使用 `--batch-id`，有歧义时须指定完整目录名。模型名中的路径分隔符会进行 URL 编码。`pilot.json` 仍是设计配置，runner 使用命令行参数和代码默认值。

每条新预测包含 `provider` 与 `answer_model`；请求响应和压缩事件记录调用服务，评分输出用 `judge_provider` 单独标识。已有旧批次的 DeepSeek 服务标识根据用户确认回填，无法确认来源的其他旧结果显示为 unknown。`provider` 标识选择的服务/密钥变量，不保存或展示 API Key。同名实验的每次运行也会自动分配独立目录。

历史批次已统一添加服务与 UTC 时间标识，例如 `runs/deepseek-flash/smoke-oracle-5--deepseek--20261116T135645680851Z/`。时间标识取自最早记录的 run 开始时间；没有可靠时间时使用 `earlier`。旧记录缺少实际接口地址，来源由用户确认并在相关元数据中注明。

如需对同一批预测使用不同评分模型，可用 `--report-id` 指定另一个独立报告目录；已有不同来源的报告不会被覆盖。

## 逐轮探针、多数据集适配与报告浏览

- **多数据集统一适配（`src/dataset_adapters.py`）**：将 LongMemEval（自动按 `answer_session_ids` 计算 `active_from_turn`）及外部/自建多轮数据集统一转换为 `turns + probes` 格式；`runner.py` 支持 `--mode unified` 直接加载标准化 JSON/JSONL 数据。
- **隔离探针与遗忘轮次定位（`src/runner.py`）**：通过 `--probe-schedule` 控制只在只读快照上运行隔离探针（不写回主历史、不进压缩器）：
  - `final_only`（默认）：仅在最终问答评估；
  - `on_maintenance`：在证据出现后、每次触发压缩/淘汰维护事件后探测，定位是哪次维护导致记忆失效；
  - `every_k_turns`（配合 `--probe-k`）或 `every_turn`：按固定间隔或每轮探测。
  - 开启探针或 `--save-snapshots` 时会保存每轮 `snapshots/turn_xxx.json`，并将探针问题、回答、预期答案、判定结果与评分原文写入 `probes.jsonl`。
- **离线报告与逐轮问答浏览（`src/report_view.py`、`src/report_template.html`）**：
  - 报告默认以「全部轮次平铺」展示每轮问题与回答、维护事件及隔离探针结果，并支持关键字过滤、探针筛选与「单轮聚焦」切换。
  - 已为 `N1–N3 / N5 / N7–N10` 集成零 API 确定性维护质量审计（`N9/N10` 含主体+值同源 `false_additions` 校验与 `fact_memory_stats` 槽位占用统计），并为 `N4` 提供 `--audit-n4` 语义维护质量审计：统一在「最终回答前（Final Readout）」共同检查点统计证据保留率（中断题标为不可评估，不计零分）、证据全齐率、首次证据遗漏/错误新增/旧事实残留，以及「丢证据 / 证据在但答错 / 执行中断」三类归因。
  - 修改报告模板后，可加 `--offline` 仅从已有评测结果离线重建 `report.html`（不调用任何 API）。

## N1 / N4 revision

N1 现为完整历史参考，不再受 `--budget 8192` 限制。所有包含 N1 的旧命令（包括默认策略列表）现在必须增加 `--n1-context-tokens`，其值为向服务商核实的模型总上下文容量；程序预留 512 tokens 给回答。不要将维护预算误当模型容量，也不要猜测 ECNU 的容量。暂不运行 N1 时显式选择其他策略。

N4 保持纯 LLM 增量压缩，使用更保守的目标长度，并在一次失败后依据原始输入缩短上次草稿；最多一次逻辑重试，不加入抽取式回退。新目标比例仍需开发集验证。

```bash
# 在 memory-maintenance-lab 中运行；此命令仅选择 N4，不需要 N1 容量参数。
.venv/bin/python src/runner.py --mode longmemeval --provider ecnu \
  --strategies N4 --dataset data/longmemeval/longmemeval_s_cleaned.json \
  --manifest data/manifests/s-dev-30.json --batch-id n4-v3-dev \
  --n4-target-ratio 0.65 --n4-retry-ratio 0.40 --n4-output-tokens 4096
```

运行 N0–N4 时，将策略改为 `N0,N1,N2,N3,N4` 并添加 `--n1-context-tokens <已核实容量>`（占位符不能直接执行）。新命令创建独立批次，不修改旧结果。配置目录中的设计文件不自动加载到 runner；以 CLI 和 manifest 的 `effective_budget` 为实际运行依据。

实验解释与限制见 [evaluation.md](docs/evaluation.md)，后续安排见 [roadmap.md](docs/roadmap.md)。离线验证：`.venv/bin/python -m unittest discover -s tests -v`。

旧开发批次已归档在 [inventory.json](archive/dev-20261117/inventory.json)，新批次写入 `runs/` 和 `reports/`。LongMemEval 运行时逐题输出进度，`progress.jsonl` 记录压缩尝试及 API usage，轮次和维护事件即时追加；中断批次可据此检查，不代表支持断点续跑。
