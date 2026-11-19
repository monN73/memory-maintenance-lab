# evaluation.md

规定数据集使用、评估口径、日志和结果展示。

## 数据集与评估协议

当前实现口径核对日期：2026-11-18。推荐以 **LongMemEval cleaned 的 S 版本为主评测**，Oracle 用于排查回答与评估链路，M 留作更长历史压力测试。原有人工场景保留为机制诊断和回归验收，不再作为主要效果结论来源。

依据：[官方仓库](https://github.com/xiaowu0162/LongMemEval)、[cleaned 数据发布页](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)、[官方评估代码](https://raw.githubusercontent.com/xiaowu0162/LongMemEval/main/src/evaluation/evaluate_qa.py)。本地已有 Oracle 与 S cleaned 数据，已完成开发子集评估。正式实验需冻结数据 revision、文件 SHA-256 和 evaluator 版本。

### 数据使用顺序

| 数据 | 项目用途 | 限制 |
| --- | --- | --- |
| 人工 `revision-smoke`（仓库初始最小冒烟样例；其他历史自建场景已归档至 `archive/dev-scenarios/`） | 仅作代码链路与格式极简冒烟，**现行正式实验已彻底停止使用自建集** | 不能代表公开基准成绩 |
| LongMemEval Oracle (`oracle-5`) | 小规模链路检查、证据充分且总长度 <16K 时的参考结果 | 已经筛选证据，不能当作真实长历史维护成绩或严格性能上界 |
| LongMemEval S cleaned (`s-dev-30` & `s-ku-10`) | 主评测（`s-dev-30` 混合题型开发集；`s-ku-10` 提前冻结的 10 道 `knowledge-update` 真实更新切片），完整历史顺序摄入后回答 | 子集须标明，不能称全量成绩 |
| LongMemEval M cleaned | 长历史与多次 compaction 压力测试 | 单独统计成本和预算，不与 S 混算 |

文件名按发布页为 `longmemeval_oracle.json`、`longmemeval_s_cleaned.json`、`longmemeval_m_cleaned.json`。不混用旧版和 cleaned 版，不默认不同变体完全同题；用 question_id 核对交集后做配对比较。官方仓库也链接了 V2，本项目先冻结这套文本历史基准，agentic 扩展作为后续独立协议。

### 两种评估协议

#### A. benchmark_final：主要效果结果

完整摄入每道题的原始 user/assistant 历史和时间信息，最后提供原始问题与 question_date。摄入阶段不提前提供最终问题，维护器按通用记忆任务维护。不得重新生成数据集中的 assistant 消息，否则变成另一套历史。

N 系列进行固定历史回放；Hermes 需实现固定消息导入/摄入，不让其自行续写替换原始历史。若必须用“历史文本作为输入”的桥接方式，保留原始 role/time 标签，记录包装 prompt，并标为 wrapped_history；与直接导入结果分组比较。已有文档中生成 assistant 回答的逐轮流程仅用于自建交互场景，不用于此协议。

不同题使用独立状态，run 的唯一键包含 dataset revision、variant、question_id、experiment_cell_id、repeat；cell 冻结策略、native profile 与编排等设置。历史 session 边界必须保留；每道题应记录是否保持单个工作上下文（continuous）或随数据 session 重置工作上下文（session_reset）。初版 continuous 为主对照，跨会话重置另设消融，不混成一个分数。

只把文本、role、session 时间交给模型；`answer`、`answer_session_ids`、消息的 `has_answer` 等评估标注单独保存。原始 question_id 用于日志关联；不能把证据标签或评估器判定注入维护器。

最终预测导出为每行 `question_id` 与 `hypothesis` 的 JSONL，一个策略/重复/变体一份预测文件；重复运行不能在同一文件制造重复 ID。当前项目报告使用本地 [eval_and_report.py](../src/eval_and_report.py) 的题型提示词和指定 judge 服务：非空回复包含 `yes`（忽略大小写）即判为正确，空回复或异常无有效判定。它沿用官方评估思路，不能仅凭生成了项目报告就声称执行了固定 commit 的官方 evaluator。正式报告应冻结所选 evaluator 或本地评估器的版本、提示词和解析规则；自建场景的严格旧值规则另算。

按原始 question_type 分组，并单列 abstention 标签；拒答是可与题型交叉的标签，统计时不能重复计入总体分母。使用指定版本 evaluator 的判定约定，额外保存原始 ID，避免靠自己猜测题型。

#### B. maintenance_trace：定位失效的诊断结果

每个完整对话 exchange 记录 source_message_indices 与 turn_id 映射；遇到不成对消息按实际边界记录，不拼造消息。逐轮快照仅在启用 `--save-snapshots` 或非 `final_only` 探针时保存；session 结束和压缩前后评估检查点属于后续设计，不是默认运行行为。

模型不会看到未来问题；只在隔离的状态副本中使用探针，不将回答和维护副作用写回主 run。最终题可作为一个诊断探针，但只能在人工确认其证据与时间条件已充分的检查点使用相应答案。知识更新/时间推理题不能把最终答案倒灌到早期状态作为标准答案。

原始证据标注适合定位来源，不等于完整事实生命周期。没有人工或审核过的补充标注时，`fact_retention`、精确遗忘轮次和 `eligible_from` 记 null。优先在 S 开发子集中预选 10 题建立证据单元、有效区间和 prefix-specific probe 标注，再扩大诊断。来源 ID 尚在不等于内容保留，来源 ID 不在也不等于摘要遗忘。

对 answerable 题，状态分为 evidence_pending、eligible、unannotated；abstention 题单独看正确拒答率，不使用“首次证据出现”计时。first_probe_failure 只有在可评估检查点才定义；从未答对属于 acquisition_failure，先答对后答错才记 observed_forgetting。后者仅说明观测到退化，归因需核对维护前后证据。

### 初批规模与预算

1. 人工样例 + Oracle 5 题检查格式与评估链路；样本 ID 预先固定。
2. 从 S 按题型与 abstention 标签固定抽取 30 题作为开发子集，写入 ID 清单及抽样 seed。先跑 N0–N4 各 1 次共 150 run，之后按需重复；该子集不用于调参后无偏泛化宣称。
3. Oracle 及人工检查暴露的同题/同历史组均计入开发暴露。参数与 evaluator 冻结后，先评估未用于开发的题集，再扩展全量 S；报告全量成绩时注明开发题有重叠，另报未用于调参的题目成绩。
4. 同一 ID 清单接入 Hermes，再扩展 M；人工细标的 trace 子集独立汇报。

当前 N1 为完整历史参考组，不受 `--budget` 的 8K 维护预算限制。必须通过 `--n1-context-tokens` 指定已核实的模型总上下文容量，扣除 512 tokens 回答预留后进行估算检查；超限记 overflow，不静默截断。旧版 8K N1 仅为容量限制开发对照。N1 与 N2–N4 不宣称等预算比较。禁止为塞进上下文而删除无关 session 后仍标为原始 S。

汇总同时显示正确数/计划题数（缺失与运行失败按未成功计，名为 end_to_end_success）、正确数/有效 judge 判定数（conditional_accuracy）、预测覆盖率和 judge 覆盖率。完整预测与有效 judge 覆盖后再报告完整基准成绩。错误分母和缺失原因必须可追溯。

### agent 原生维护激活补充（当前暂缓）

> **阶段说明**：当前评测聚焦于 `direct runner` 下的外部记忆维护方法参考 Baseline（统一采用 `N0–N14` 纯数字编号），Hermes 等 Agent 原生维护激活测试暂缓执行。

固定历史回放保留数据不变，但不保证触发 agent 的自然记忆写入。后续启动 Agent 对照时，按[agent-integration.md](agent-integration.md)记录 ingestion_activation 并验收原生事件。30 题 × 5 策略仅指最初 direct 实现检查；原生机制交互单独按 cell manifest 规划题数，不与该 150 run 混算。

时间排序、session reset 与辅助模型预算按[agent-integration.md](agent-integration.md)和[research-protocol.md](research-protocol.md)执行。

## 指标与逐轮记录

### 通用指标与设计目标

设 S(x) 为 Unicode NFKC 标准化、去空白后文本的字符二元组集合；保留数字、标点、大小写。中文不依赖空格分词。标准化算法版本为 `nfkc-char-bigram-v1`；长度不足 2 的非空文本用单字符集合。

| 指标 | 定义 | 用途及边界 |
| --- | --- | --- |
| lexical_jaccard | \|S(before)∩S(after)\| / \|S(before)∪S(after)\| | 同次维护输入/输出的字面重叠；摘要改写后低分不代表遗忘 |
| lexical_retention | \|S(before)∩S(after)\| / \|S(before)\| | 原文本片段保留程度，非语义正确率 |
| adjacent_memory_overlap | Jaccard(M上一轮, M当前轮) | 记忆稳定性；高分也可能是没有写入新事实 |
| duplicate_ngram_rate | 1 − 唯一二元组数/二元组总数 | 冗余代理，不能判断语义重复 |
| fact_retention | 正确保留的当前有效事实数/应保留的当前有效事实数 | 分别测 M 和真正可见的 C，区分存储与注入问题 |
| stale_fact_rate | 当前仍出现的已失效事实数/截至本轮已失效事实数 | 历史值若明确标为历史，不算 stale |
| unsupported_fact_rate | 无来源支持的事实条目数/输出事实条目数 | 观察压缩引入的幻觉，不能只靠字符串匹配 |
| probe_accuracy | 正确探针数/有效探针数 | 结果正确性，按事实/历史/组合/未知分别统计 |
| compression_ratio | output_tokens/input_tokens | 越小压缩越强；统计对象为被维护片段，不能混入未变的近期历史 |
| update_lag | 首次正确替换旧值的轮次 − 修订轮次 | 分 M 和 C 测；至结束未更新记 censored |
| deletion_lag | 首次移除/标记撤销的轮次 − 撤回轮次 | 若之后复活，额外记录 resurrection 事件 |

通用规范是分母为 0 时记 null 并说明原因；现有实现例外见下文。没有对应事件时不补零。预算淘汰和 N7 检索也可能产生事件长度比，不能将这些全部解释为摘要压缩率。没有 M 的策略不应用 M 的指标，但仍可检查实际可见上下文。

### 现有报告：证据保留率

**先看分母。** 现有问答报告里的证据代理是 `answer_session_ids`：把属于答案会话的所有原始 exchange 都列入候选，包括其中与答案无关的轮次。按原始消息顺序组装 exchange；同轮的 user 和 assistant 共用一个 `turn_id`。离线回放使用更窄的 `has_answer` 消息标注，见下节。两套指标都不能替代逐事实语义标注。

最终检查点统一定义为**最终问题完成预算处理后、模型回答前**。每题先算「保留单位数 ÷ 应检查单位数」，再将有值的题等权平均（`avg_evidence_retention`）；`evidence_evaluable_count` 是参与平均的题数。`all_evidence_retained` 表示该策略口径下所有单位均保留；总览另计其为 true 的题数。不要按维护事件数加权。

| 策略 | 单位 | 怎样算保留 |
| --- | --- | --- |
| N1、N5 | 答案会话的原始轮次 | 完成且有候选证据时为 100%，因为完整回放；这不证明模型成功找到了证据。 |
| N2 | 答案会话的原始轮次 | 统计仍在近期窗口的候选轮次。 |
| N3 | 按原始轮次分别去重的规范化句子 | 仍在近期窗口的轮次计入全部句子；已淘汰轮次只计维护输出中有相同来源 `[T<id>]`、且规范化后相同的句子。按常见标点或换行断句，转小写并折叠空白，与字符二元组重叠算法不同。事件卡片同时显示保留轮数（匹配到任一句就计一轮）；它不是保留比例的分子。当前实现若有证据轮但无可分句文本，比例回退到 100%。 |
| N7 | 答案会话的原始轮次 | 摄入阶段的事件只看近期窗口；最终统计「近期窗口 ∪ BM25 已召回轮次」。写入归档但未召回不算最终可见，也不等于数据被删。 |
| N8、N11、N12 | 答案会话的原始轮次 | 与 N7 相同，但摄入阶段若候选轮次因归档超出 `--n8-archive-tokens`（默认 16K）分别按 `N8`（先进先出）、`N11`（精确去重后先进先出）或 `N12`（会话均衡淘汰）被淘汰（记录在 `archive_evicted_turn_ids`），则从可召回集合中扣除；最终统计「近期窗口 ∪ 有界归档 BM25 已召回轮次」。 |
| N9、N10、N13、N14 | 答案会话的原始轮次（结合 `[T<id>]` 事实条目来源追踪） | 仍在近期窗口的候选轮次或维护/检索输出 `M` 中保留对应 `[T<id>]` 条目的候选轮次计为保留（`N13` 合并统计结构化事实与有界归档 BM25 召回轮次），并辅以 `fact_memory_stats` 统计槽位占用与覆盖命中率。 |
| N4 | 答案会话的原始轮次 | 有 `audit_N4.jsonl` 时才展示。原文仍在近期窗口直接计入；被淘汰轮次由 LLM 判断最终摘要能否支持核心事实，合理改写可计入。 |
| N0、N6 | — | 当前报告未实现对应证据审计；空白不表示零（N6 因辅助 API 成本过高暂缓运行）。 |

确定性审计依据 `events.jsonl` 推断保留状态，不逐条解析最终请求。事件级分母只用 `turn_id` **小于**事件轮次的历史证据；不能据此定位当前 exchange 刚摄入时的语义失效。确定性审计（[report_view.py](../src/report_view.py)）当前处理 `sliding_window_eviction`、`extractive_compaction`、`bm25_index_eviction`、`bm25_bounded_eviction`、`bm25_dedup_eviction`、`bm25_session_balanced_eviction`、`bm25_retrieval`、`fact_append_compaction`、`fact_update_compaction`、`fact_hybrid_compaction`、`fact_enhanced_compaction`，没有纳入 `target_unreachable_eviction` / `zero_memory_budget_eviction`。遇到这些分支，应以原始事件及 `requests/final_answer.json` 为准核对可见证据。执行中断或没有候选证据时，最终比例及“全齐”为空；仍可能有已经记录的部分事件计数。

### 离线证据轨迹与快照重放口径（`export_evidence_trajectory.py` & `probe_from_snapshots.py`）

为保证 LongMemEval 过程追踪与重放验证的无偏与可复现，离线脚本严格执行以下五项口径：

1. **双证据层与双分母分离**：`has_answer` 用于消息级证据单元定位（映射到 exchange 时保留 `source_message_indices`）；`answer_session_ids` 为更粗的会话级代理。两者在 `evidence_trajectory.md/.json` 中分列、分别计算分母；当某题缺少消息级 `has_answer` 标注时显式记为 `missing`，绝不静默回退成整个 `answer_session_ids`。
2. **三层独立追踪与 `N11` 去重区分**：分别检查**近期窗口（`recent_window`）**、**外部存储（`external_storage`）**、**探针读出（`probe_readout`）**。`N11` 发生精确去重时，同步区分支源轮次（`source_turn_id`）保留与规范化正文内容保留（记录实际承载轮次 `via_turn_id` 与时间戳 `timestamp`）。
3. **严格区间与事件归因边界**：`observed_loss_interval` 固定报告 `(last_observed_retained_turn, first_observed_missing_turn]`。`N7/N8/N11/N12/N13` 的窗口淘汰（`evicted_turn_ids`）仅代表证据迁入归档，只有对应存储层的淘汰事件（如 `archive_evicted_turn_ids` 或 `[T<id>]` 从 `M` 消失）才构成该层遗失归因；区间无匹配事件记 `unattributed_in_interval`，多事件记 `ambiguous_multiple_events`，`N4/N6` 改写记 `uncertain_llm_paraphrase`（不确定从缺失推断遗忘）。
4. **快照重放权限与一致校验**：重放权限表包含 `N5`；`N0–N6`、`N9–N10`、`N14` 严禁读取历史原文归档，`N7/N8/N11/N12/N13` 仅允许读取快照冻结的 `archived_turn_ids` 白名单（越界则报错），`N13` 同时恢复 `memory_text` 事实表与 `archived_turn_ids` 白名单。在线与事后 probe 共用 `build_readout_memory` 与 `build_messages`，`--verify-only` 逐条核对 `retrieved_turn_ids`、`memory_text_injected` 与规范化请求哈希 `request_sha256`（验证模型收到**完全相同输入**）。
5. **开发暴露样本说明**：`s-ku-10` 中的 `01493427` 与 `45dc21b6` 用于链路重放验证及 `N14` 前端提取规则检查，报告与统计分析时单列标明为公开切片内的开发暴露样本。

### 现有报告：遗漏、新增、旧值与失败分类

| 字段 | 实际算法 | 解读限制 |
| --- | --- | --- |
| `fact_omission_count` | N2 统计本次首次移出窗口的候选证据轮；N3 统计首次匹配不到的「来源轮次、规范化句子」对；N7/N8 统计摄入时首次移出窗口（或归档淘汰）的候选证据轮，最终 BM25 事件不增加此计数；N9/N10 统计首次从近期窗口和事实槽位中均消失的候选证据来源。N4 审计跟踪首次不再由原文或摘要支持的候选轮次。 | 同一单位首次遗漏只计一次；N7 此处是窗口迁出，不代表永久删除，也不等于最终遗漏量。 |
| `false_addition_count` | N2/N3/N7/N8 用维护后规范化句子集合减去维护前集合；N9/N10 使用 `_count_fact_false_additions` 剥离 `[T<id>]` 与 `ent \| attr =` 前缀，并要求主体实体 `ent`（非通用 `user` 时）与值 `val` 必须可溯源到维护前的同一条源句（防止把“杉木的负责人是周”错写成“海盐的负责人是周”时因右侧值相同而漏检）；N4 由审计裁判判断摘要中缺乏输入支持的断言数。 | 字面/同源差集不是完整语义事实判定，跨策略解读须结合机制说明。 |
| `stale_fact_residue_count` | 确定性分支仅在 `knowledge-update` 且已有至少两个候选轮次时，把早于最新候选轮且仍在近期窗口或事件输出中的候选轮计入；N4 裁判判断是否把旧值当作当前有效事实。可控场景另通过 `fact_memory_stats` 直接统计 `duplicate_key_tokens` 与 `retracted_tokens`。 | 确定性分支不理解自然语言时效，正常历史也可能命中。总览的 `total_stale_residues` 是各题最后一次已审计维护事件的计数之和，不是每个事件累加，也不总是实际最终请求中的旧事实数。 |

总览的 `total_fact_omissions` 和 `total_false_additions` 是各题事件计数之和，会受事件数、方法与审计覆盖影响，不是率或标准化排名。N4 可用 `--audit-n4` 发起额外 LLM 审计，也可复用缓存；`--offline` 只复用已有结果。需检查 `audit_error`：裁判调用或解析失败时，当前实现可能仍保留数值，但无法验证的裁剪证据不会计为已保留，不能把该数值当成可信的语义分数。未开启或没有缓存时留空。

结果标签按顺序判断：未完成为 `execution_failure`；答对为 `correct`；答错且证据不齐为 `evidence_lost`，答错且证据全齐为 `evidence_present_wrong_answer`，答错但无证据审计为 `wrong_no_evidence_eval`；完成但无有效评分为 `unjudged`。这些标签描述共同出现的结果，不证明证据丢失造成答错。事实级保留、遗漏、新增和旧值仍需有效期和语义标注，见本文件的通用指标定义。

### N7 离线回放：存储与读取分开计算

[replay_archive_maintenance.py](../scripts/replay_archive_maintenance.py) 复用原 N7 批次的写入事件，固定近期窗口、最终问题、BM25 读出规则和回答窗口，只改变归档写入与淘汰策略。原 N7 的归档没有容量上限；离线对照单独设置归档上限，不改变回答输入预算，不调用 API 生成回答或评分。回放按原始数据 **`has_answer: true` 的消息** 定义证据，映射到 exchange 后按轮去重；一个轮次中有两条标注也只计一次。

| 指标 | 分子 / 分母 |
| --- | --- |
| 维护检查点证据保留 | 到该轮为止已出现（含当前轮）的标注轮次中，仍在近期窗口或归档中的数量 ÷ 已出现的标注轮次数。`checkpoints.jsonl` 保存原始计数；没有标注出现时不计算比例。 |
| 读前证据保留 `stored_evidence_recall_macro` | 每题仍在「最终保留的近期窗口 ∪ 参与读取的归档」中的标注轮次数 ÷ 该题全部标注轮次数，再按有标注题取宏平均。 |
| 最终可见证据 `visible_evidence_recall_macro` | 每题进入「最终保留的近期窗口 ∪ BM25 选出的原文」的标注轮次数 ÷ 同一分母，再按有标注题取宏平均。 |

例如一题标注 4 轮，最终近期窗口保留 1 轮、归档另保留 2 轮，BM25 只读出其中 1 轮，则读前保留为 3/4，最终可见为 2/4：1 轮已不在状态，另 1 轮仍在归档但未被读出。无标注题不参与平均。归档大小以逐条序列化后估算 token 数之和计，不是物理内存或 API token 用量。

产物保存在 `diagnostics/archive-maintenance/`：`summary.json` 记录来源批次、数据 SHA-256、有效预算、归档容量以及与原始 N7 请求匹配的题数；`question-results.csv` 和 `checkpoints.jsonl` 可按题、策略和维护轮次核对分子分母。现有产物未冻结回放脚本哈希，作为开发诊断单独解读，不写入问答报告的成绩总表。策略和预算细节见 [research-protocol.md](research-protocol.md)。

### “什么时候失效”的操作定义

- first_fact_loss：对某个此前已正确保留、仍有效且需要保留的 fact，首次在 C 中无法正确恢复；M 中丢失单独记录。
- first_probe_failure：同一事实的探针第一次答错的轮次。
- sustained_failure：同一事实连续 3 次有效观测答错，事件起点为这三次中的第一次；同时保存确认轮次。缺失探针不算答对或答错，报告实际观测间隔。
- stale_answer：回答使用已被覆盖/撤回的值。
- overflow：输入超预算且该策略无法继续；独立于记忆答错。
- infrastructure_error：API、超时、解析或适配器失败，不计作认知失效。

正常观察结束时仍未失效记 right_censored=true 和最后观测轮次；overflow 是竞争终点，API 中断独立标记，不作普通右删失，不能将它当作“在最后一轮失效”。低频探针只能报告上一次正确与第一次错误之间的区间，不能声称定位到准确遗忘轮次。

跨 run 汇总：报告完成数/总数、探针准确率、事实保留率、首错轮次分布、截至轮 t 未失效比例、维护次数与成本。按 scenario 配对比较；置信区间按 scenario 重采样，避免把同一轨迹每轮当成独立样本。首错均值若只算失败 run 会有偏差，须保留删失信息。

### 记录布局

```text
runs/<模型>/<批次>/runs/<run_id>/
  manifest.json              # 完整生效配置、版本、时间、状态、数据与 prompt hash
  turns.jsonl                # 每个规范化 exchange 一行，含 source message 映射
  events.jsonl               # 追加/修订/删除/维护/重试等事件
  progress.jsonl             # 阶段、重试与用量事件（视运行模式而定）
  probes.jsonl               # 启用探针时记录输入、答案、判定和辅助 usage
  snapshots/                 # 启用快照时保存可观测状态
  requests/                  # 每次主/辅助模型实际提交消息及响应
  artifacts/                 # Hermes 原始轨迹和状态快照
  summary.json               # 汇总，未实现前不伪造
```

统一索引为 run_id、scenario_id、session_id、turn_id、event_id、request_id。Hermes 一个用户轮可能含多个模型和工具步骤，额外记录 step_id。文件路径相对 run 目录，内容 hash 使用 SHA-256。

`templates/turn-record.json` 是字段模板，null 是未采集，不是 0。生产日志还需引用 events/probes 和完整请求；不得只保存摘要后的文本。usage 区分 answer/maintenance/evaluation，并分别保存 provider 实际值与本地预估值；成本只在有模型价格快照时计算，否则 null。

对于 Hermes，持久文件内容、会话注入快照、当前实际请求是三个不同观测对象。记录 memory write 成功/失败和 session 边界，不能看到磁盘有事实就认定模型本轮可见。

### LongMemEval 适用范围补充

上述事实指标要求额外的有效期和语义标注；原始证据标签不能直接构造完整事实保留率。未标注时记 null，区分来源追踪率与语义保留率。最终准确率按已冻结且注明身份的 evaluator 计算；人工场景的严格规则另存 diagnostic_label，不覆盖问答判定。

记录新增 dataset_name/revision/variant、question_id/question_type、is_abstention、protocol、subset_manifest、ingestion_mode、session_policy、source_message_indices。checkpoint 另存 evidence_eligibility、expected_answer_version、last_correct_checkpoint、first_incorrect_checkpoint。仅 eligible 且探针有效期相同的观测可判断退化；先前未答对称 acquisition_failure，不能称遗忘。

公开基准按 question_id 配对统计与重采样；最终正确/计划数、正确/有效判定数及两种覆盖率同时报告。具体口径见本文件的数据集协议与结果展示部分。

### 原生与外加策略归因

run manifest 新增 strategy_spec_id、agent/version、native_profile_id、integration_mode、replacement_scope、ingestion_activation、capability_audit 和 experiment_cell_id。每次事件使用 [maintenance-event.json](../templates/maintenance-event.json) 记录 owner、mechanism、phase、输入输出与父事件。准确率相同也需检查原生恢复是否掩盖外加策略损失；细则见[agent-integration.md](agent-integration.md)。

交互统计使用同 agent 内配对四格，分别计算准确率和成本的交互项。不完整单元、配置失配、未知控制能力不进入受控估计，单独报告数量与原因，不能静默删除。

### v0.2 统一约定

一个轮次可能有多个 native/external 维护事件，lexical_jaccard、lexical_retention、compression_ratio 的权威值位于 events.jsonl；turn 中对应字段仅在恰有一个事件时镜像，否则 null + multiple_events。M 指标必须有 memory_scope（external/native/combined），不可悄悄混合。

全项目采用 turn_id 表示全局 exchange 序号，source_message_indices 为该 exchange 的原始消息序号列表，source_turn_id 保留原场景编号。旧 global_turn_id/source_message_index 为 v0.1 别名，不在新日志中混用。摄入阶段没有真实模型回答时 answer_text=null，原始 assistant 内容保存在输入消息引用中。

run 身份、主指标分母、竞争终点、抽样与重复规则以[research-protocol.md](research-protocol.md)为准；探针读出模式以[agent-integration.md](agent-integration.md)为准。

## 结果展示

现有 direct 报告为离线 HTML + CSV/JSONL，不需要服务端。下文同时记录当前行为和后续展示设计；证据热力图、agent 四格及精细事实生命周期视图属于设计，不代表已有实现。人工 CSV 仅用于演示，不能进入基准汇总。

### 1. 顶部筛选与结果身份

显示 dataset/revision/file hash、variant、subset ID 与题数、protocol、harness、strategy、模型、预算、重复次数、judge/evaluator 版本、摄入方式和 session policy。解释差值前必须核对数据清单、协议、预算和 judge；跨预算展示应明确分栏。

三个视图分别为“基准总览”“单题维护轨迹”“人工机制诊断”。benchmark_final 与 maintenance_trace 不共用一个准确率。未运行显示“待运行”，缺少标注显示“未标注”，空值不能渲染成零分。

### 2. 基准总览：先看最终效果与成本

| 策略 | harness | 数据/题数 | 预算 | 预测/判定覆盖 | 正确/计划 | 正确/已判定 | overflow/API/judge 错误 | 维护次数 | token/成本 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 待运行 | — | — | — | — | — | — | — | — | — |

“正确/计划”表示端到端成功率，“正确/已判定”表示条件准确率；完整覆盖才可展示完整基准分数。各类错误单独计数，不能让失败 run 从表格消失。完整分数旁显示 judge、样本量和置信区间；按 question_id 配对重采样（共享历史时以 history_group_id 聚类），重复运行先按题聚合，不把每轮当独立样本。

当前实现补充：

- `planned` 来自批次内该策略的 run 记录数，`predicted` 按 `status=completed` 计，`judged` 按有效判定计。`scoring_pending` 是已完成却没有有效判定的题（含 judge 错误）；它与运行失败分栏。若批次曾中断，需把记录数与子集 manifest 对照，避免将部分批次误读为完全覆盖。
- `total_tokens` 汇总回答和维护模型的输入、输出 usage，不含探针、评分及 N4 证据审计开销；`final_ctx_tokens_est` 是本地估算。`total_maint_events` 是被记录的事件数，N7 的问前检索也计入，不能当作 LLM 调用次数。
- `readout_latency` 是检索加回答；`total_latency` 再加已记录的维护耗时，不代表含摄入、服务排队和裁判的整批墙钟时间。N7 的检索耗时单列，维护耗时字段不衡量本地归档 CPU 成本。
- 现有配对差值取两策的公共 question_id，只把有效答对编码为 1，其余为 0；按题做配对 bootstrap。应核对数据、预算、judge 和两边覆盖率，尤其留意待评分题导致的暂定差值。

计划扩展的图表：

- 分题型准确率及样本数；abstention 另做筛选/面板，避免重复计算总体。
- 准确率与总 token/费用散点图，颜色区分 harness，点标签给出策略和预算；费用未知则不绘费用轴。
- 历史累计 token、session 数和维护次数分桶后的成功率，每桶展示题数。
- 策略 × question_id 成败矩阵，点击题目进入配对详情；题目顺序固定，避免排序掩盖失败模式。

Oracle、S、M 分页。Oracle 显示“证据筛选诊断”，不是默认最优策略；full-context 大预算参考独立于受限预算组。

### 3. 单题详情：定位哪一步开始出问题

题目头部显示 question_id、question_type、abstention、question_date、最终问题、标准答案、模型答案、原始 judge 输出。标准答案仅在评估展示中可见，不进入实验 prompt。可选择两种策略并排查看同一题。

时间轴支持切换 turn_id、session、累计摄入 token；保留原始 session_id/source_message_indices 和时间戳。计划中的主曲线包括请求 token 与预算线、M 大小和维护压缩比；当前曲线使用估算上下文 token，lexical overlap 只作为辅助。非维护轮用断点，不补零。

在时间轴上标出：证据首次摄入、session 边界、压缩/淘汰、记忆写入、更新/撤回（仅已有标注时）、探针检查点和运行错误。将“累计摄入的历史量”与“实际模型上下文大小”画为不同曲线。

支持点击事件查看：

1. 原始相关消息与来源索引。
2. 维护输入、输出和文本 diff，标出被删除/改写的证据片段。
3. 当前持久 M、实际注入块和最终请求，区分写入与可见。
4. 检查点探针、该前缀的有效答案、正确性与 judge/人工证据。
5. 后续能否恢复，以及是否访问了额外历史来源。

当前 LongMemEval 默认 `final_only`，不能用最终答错推断首次遗忘。启用中途探针时，现有页面用首条 `correct === false` 记录显示“首次观测失败”；没有有效判定不能据此声称从未失效。正式失效分析应报告“最后正确检查点 → 首次错误检查点”区间，正常观察结束未失败记右删失，overflow 和运行中断分别处理。

### 4. 证据热力图：按标注能力展示

基础模式以证据来源 session/message 为行、检查点为列，只表示来源追踪状态，不冒充语义保留率。语义模式仅对补充标注的 evidence unit / fact 开启，显示在 M 和 C 中的保留/丢失/过时状态。

颜色和标签包括：尚未出现、已保留、丢失、过时、已撤回、未标注、未评估。必须同时有文本标签或图例，不能只靠颜色。原始数据没有完整 fact 生命周期时，隐藏 fact-retention 总分，显示“需要补充标注”。

诊断卡片分别显示：

- `acquisition_failure`：达到可评估条件后尚未观察到答对（对应从未正确写入 `M`/归档的漏写 `acquisition_omission` 或初始获取失败）。
- `observed_forgetting`：此前已正确写入并在前序有效检查点保留/答对，之后在仍有效且同一期望值的探针上答错（对应有效事实遗忘 `valid_fact_forgetting`）。
- `intended_revision_removal`：按显式更新或撤回指令删除、覆盖或以 `[RETRACTED]` 失效化不再有效的旧值——属于**正确维护行为**，不得计为遗忘或事实丢失。
- `stale_answer`：回答使用已被替代或撤回的旧信息（需时效标注；即应当发生 `intended_revision_removal` 而未能生效）。
- `answer_failure_with_visible_evidence`：证据在实际请求中但回答错误。
- `unknown`：缺失快照或证据不足，无法归因。

> **跨策略共同检查点与失效区间规则**：不同策略触发维护的轮次不同（各自第 $k$ 次维护事件不能跨策略对齐）。配对比较“同一事实经过不同策略后发生了什么”时，必须对齐到**相同历史前缀的 `turn_id` 或 session 末共同检查点**，逐事实记录“是否写入、预期更新/撤回、共同检查点状态、同探针读出”四项；仅观测到前后离散点变化时，报告失效区间 `(last_ok_turn, first_fail_turn]`。目前 `s-ku-10-clean`（`N7–N10`）与 `s-ku-10-merged5`（`N4 + N7–N10`）在最终问答口径上已完成并可用，现行及后续过程追踪统一直接在 LongMemEval（如 `s-ku-10` 含 `has_answer` / `answer_session_ids` 位置的共同检查点）等公开基准上落盘快照并离线抽取对比。

重叠率只回答文本改动程度，不能用高/低 overlap 直接给记忆“正常/失效”的红绿灯。完整基准最终答错而无检查点时，仅显示“最终答错，失效时点未知”。

### 5. 导出与验收

- `predictions_<策略>.jsonl`：每个策略/重复/变体一份，保留官方 evaluator 所需字段。
- `question-results.csv`：逐题结果，至少含数据身份、策略、预算、判定、错误状态和成本。
- `checkpoints.jsonl`：检查点位置、eligibility、前缀答案版本、探针结果与快照引用。
- `turns.jsonl`、`events.jsonl`：全部摄入轮次及维护事件，不因探针稀疏而省略轮次日志。

点击总览的一项失败，应能追溯到题目、历史消息、维护事件和实际上下文；无法确认原因时明确标 unknown。数据变体、子集、预算或 judge 不一致时不可合并排名。没有真实 run 时保留空态，不填造分数。

### 6. 保留的人工展示样例

[illustrative-timeline.csv](../examples/illustrative-timeline.csv) 展示“负责人更新后，压缩使旧值复活”的假想过程，仅归入人工机制诊断视图。它没有 LongMemEval question_id，不得标成公开数据样本；高 overlap 与答错共存的演示用途保持不变。

### 7. 计划：策略 × agent 原生机制视图

顶部增加 strategy_spec、native_profile、integration_mode、replacement_scope、ingestion_activation 与有效能力状态筛选。显示 agent 内 A/B/C/D 四格，分别列出准确率、覆盖率、总 token 和耗时；仅在四格完整且控制条件相同时展示 B−A、D−C 与交互项及置信区间。公式和限制见[agent-integration.md](agent-integration.md)。

策略可以来自写入、表示、更新、压缩、淘汰和读取维度，不能把页面列固定成 N0–N4。每种 agent 独立展示 native_only 基线、replacement 与 coexist；跨 agent 排名只作为系统级比较，不宣称纯维护因果效应。

单题轨迹将 external 与 native 事件分泳道，用 parent_event_ids 连出实际处理顺序。点击某次失败可看到：外加策略输出是否仍含证据、原生二次处理是否删除、历史搜索是否恢复、最终上下文是否重复或存在新旧值冲突。

新增诊断卡：双重压缩次数、原生恢复事件数、重复注入占用、版本冲突数、预算开销拆分。没有语义标注或日志支持的数值记 null；检测到双重压缩不直接等于它造成失败。配置上关闭但运行时仍触发时标 configuration_mismatch，该 run 不进入预期单元的受控比较。

显示可控性/观测性矩阵：已验证开、已验证关、未知、不支持、未激活。被禁用能力的日志证据可点击查看，不能用“本次没触发”替代关闭验证。

### 8. 计划：比较有效性与编排视图

优先展示 comparison_id、claim_scope、changed_factors、orchestration_profile_id、identity 控制是否匹配、capability 审计状态和样本覆盖。将 controlled_component / method_bundle / system_observational 分栏，不输出混合排行榜。

四格区分 quality 与 reliability：容量失败较多时突出 overflow/覆盖率，不将质量交互画成已识别结论。单元不完整或 wrapper 混杂显示 not_identifiable/wrapper_confounded。另展示无 wrapper、identity wrapper、相同策略不同 wrapper 的对照，区分调度开销与维护收益。

探针显示 snapshot_readout/operational_probe，事件级指标以 event_id 为横向细分，单轮多事件不被一个 overlap 值覆盖。报告开发暴露、未用于调参的题集、history_group_id、重复次数和置信区间。分母与终止分类遵循[research-protocol.md](research-protocol.md)。

### 诊断读出与主实验的界限

snapshot_readout 的同一前/后对照使用相同 reader、prompt、输出预算和输入硬上限；先为探针文本预留固定空间。完整快照与探针放不下时记 probe_overflow，不默默裁剪、不临时扩窗，也不把它判作事实丢失。必要时另建较大窗口诊断组，明确它不属于等预算主成绩。

诊断曲线必须同时显示可评估检查点数、预算失败数和维护暴露次数。按观察到的错误区间加密探针属于探索性定位，不能与原先稀疏检查点一起伪装成预定的首次失败统计；保存原计划与追加观测标识。

LongMemEval 最终问题正确性衡量任务效用；细标事实子集的保留率只代表该标注范围，不代表历史中所有信息。原生历史恢复使最终答案正确时，同时报告存储损失和恢复事件，避免将系统纠错误判成维护无损。

每次运行按 `runs/<模型>/<batch-id>--<生成服务>--<带时区时间戳>/` 保存，报告按 `reports/<生成模型>/<批次>/` 保存；文件内容记录生成与评分服务。旧记录缺少来源时标记 `unknown`，不得仅凭模型名推断密钥来源；已经由用户明确确认的旧结果可回填并标记确认来源。

## N1 / N4 implementation revision

这两项在 v3 引入；当前批次的实际执行器版本见 manifest 的 `budget_executor_version`。研究主线是相对完整历史的效果损失与资源节省，不能将执行失败等同于事实遗忘。

- N1：完整保留历史；CLI 必须显式提供模型容量，不根据服务名称猜测。容量检查采用 `approx-utf8-v1`，不是服务商 tokenizer 的精确计数，API 仍可能拒绝请求；记录该失败，不能偷偷删历史。
- N4：保留 N3 的预算分配、阈值和历史访问范围。第一次摘要目标为记忆预算的 0.65，失败后目标为 0.40，最多两次逻辑调用；第二次同时提供原始维护输入和上次草稿，要求核实后缩短。没有抽取式或截断回退。
- `--n4-output-tokens` 默认 4096，是 API 生成上限；保留记忆预算默认 2048 个估算 tokens，两者不是同一计量口径。目标比例可用 `--n4-target-ratio` / `--n4-retry-ratio` 配置，应在开发集校准后冻结。开发批次效果不能替代正式冻结与验收。
- 空输出、截断、超预算在重试后仍失败则终止该题并记录原因；API 错误沿用客户端传输重试，不能与摘要缩短次数混淆。维护不访问标准答案或未来历史。
- manifest 保存 `effective_budget`、执行器版本、方法策略标识和提示词 hash；事件保存每次目标长度、生成上限、请求、响应、预算检查结果。

报告同时呈现全部样本效果、完成率、各类失败及资源用量。完成样本准确率存在选择偏差，只作辅助；超预算输出可用于诊断，不混入合规记忆结果。旧批次保留原记录，标为开发结果，不与新版混算。

LongMemEval 回放的预算检查已计入当前完整 exchange；轮次和事件实时追加，压缩调用的开始、结束及 usage 记录在 progress.jsonl。预算仍使用近似 tokenizer，正式比较前应完成在线校准。历史开发批次位于 archive/dev-20261117，不纳入新版比较。

### LongMemEval 逐证据维护轨迹

按实际快照追踪近期窗口、外部存储和探针读出。分别报告 has_answer 消息数、它们对应的 exchange 数，以及 answer_session_ids 会话代理覆盖的轮次数。源标记消失只能支持来源级判断。摘要语义保留需要独立审核。N11 另保留等价正文的替代来源与时间戳。

重放只使用当时快照中的近期原文、记忆和合法归档白名单。验收必须同时核对在线记录的请求哈希、记忆正文和召回 ID。标注只用于事后评估。前缀答案缺少时效标注时，探针可以记录读出，评分仅在有独立有效答案时解释为准确率。

方案 A 的报告目录与主批次同名。先生成证据轨迹，再生成 HTML，使报告页面链接到轨迹文件。尚未评分时，可用 `--unscored` 生成完成率、成本和证据报告。待评分题不计为判错。

```bash
.venv/bin/python scripts/export_evidence_trajectory.py --batch-dir "$batch_dir" --dataset data/longmemeval/longmemeval_s_cleaned.json
.venv/bin/python src/eval_and_report.py --batch-id "$batch_id" --dataset data/longmemeval/longmemeval_s_cleaned.json --unscored --report-id "$batch_id"
```
