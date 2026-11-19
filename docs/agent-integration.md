# agent-integration.md

规定外部记忆维护策略的可组合维度拆解，以及**暂缓执行**的 Agent 原生记忆对照、外部编排边界和 Hermes 接入验收协议。

## 策略与原生机制对照

### 研究定位与暂缓接入说明

1. **当前核心主线（`direct runner` 外部记忆维护参考 Baseline）**：
   - 当前研究聚焦于 **“不同的外部记忆维护方法会造成什么影响”**，通过可组合的 8 个策略维度（摄入粒度、写入选择/前端提取、表示、合并与更新、删除与淘汰、压缩算子、触发条件、读取与注入）构建覆盖 `N0–N14` 纯 `N + 数字` 编号的参考 Baseline 体系。
2. **为什么当前暂缓 Agent（Hermes）接入测试**：
   - 真实 Agent 框架（如 Hermes）自带系统提示词开销、工具 schema 预算挤占、`MEMORY.md` 会话起始冻结快照以及 agent/gateway 双层上下文压缩。若在外部维护基线尚未完备时直接接入 Agent，会产生严重的包装混淆（Wrapper Confounding），难以区分结果变化是源于“外部记忆维护算法本身”还是“Agent 框架包装与原生机制的副作用”。
   - 因此，**Hermes 接入及同 agent 内 `A/B/C/D` 四格实验明确暂缓**，优先在零干扰的 `direct runner` 中完成外部记忆维护基线矩阵（含同容量有界归档多淘汰规则 `N8 / N11 / N12` 与结构化事实前端提取/混合读取 `N9 / N10 / N13 / N14`，所有策略均采用独立的 `N + 数字` 编号，不使用带连字符的子编号）。本文档中的 Agent 原生能力审计与编排契约作为后续二期研究的规范储备保留。

原来的 N0–N6、H0–H4 只是起步预设，当前 direct 主线统一使用 `N0–N14` 纯数字编号。Hermes 是预留的第一个 Agent 适配对象；以后接入其他 agent 使用相同能力清单，不假设它们具有相同配置接口或行为。

### 1. 将策略拆成可组合的维度

| 维度 | 可选方法示例 | 主要观察 | 对应 Direct Baseline 策略（`N0–N14`） |
| --- | --- | --- | --- |
| 摄入粒度 | 消息、完整轮次、session、固定 token 块、主题块 | 边界丢失、维护频率 | `N1`（全量） vs `N5`（2K 分块原序重组） |
| 写入选择 / 前端提取 | 全量、最近优先、固定规则提取、扩展语义提取 | 选择偏差、自然语言事实漏提（`99.25% fallback`） | `N9/N10`（基础规则提取） vs `N14`（增强前端提取） |
| 表示 | 原文摘录、自然语言摘要、`(entity, attr, value)` 条目、结构化事实+原文混合 | 精确信息保留、结构化开销 | `N3`（句摘录）、`N4`（LLM 摘要）、`N7/N8/N11/N12`（原文）、`N9/N10/N14`（事实条目）、`N13`（事实+原文混合） |
| 合并与更新 | 只追加、精确去重、同键覆盖更新、带显式失效标记的版本记录 | 新旧值冲突、重复项挤占预算、旧值残留 | `N9`（只追加） vs `N10/N13/N14`（同键更新覆盖） |
| 删除与淘汰 | 滑动窗口 FIFO、归档 FIFO、精确去重后 FIFO、会话均衡淘汰、显式 `[RETRACTED]` 撤回 | 早期关键事实被淘汰、撤回未生效、单一长会话刷爆归档 | `N2`（窗口淘汰）、`N8`（归档 FIFO）、`N11`（归档去重 FIFO）、`N12`（归档会话均衡淘汰）、`N10/N13/N14`（显式撤回与恢复） |
| 压缩算子 | 无、字面抽取、单次/增量 LLM、分块分层 LLM | 累积改写损失、幻觉、API 延迟与 Token 成本 | `N3`（抽取） vs `N4`（增量 LLM）；`N6`（分块 LLM，因高 API 成本暂缓批测） |
| 触发条件 | 每轮、token 阈值（`>0.8B` 压缩至 `≤0.6B`）、查询读出触发 | 维护频率与资源消耗 | `N2–N4/N7–N14` 统一共享 `0.8B / 0.6B` 触发契约 |
| 读取与注入 | 全量 M、BM25 稀疏检索、按实体/属性匹配、结构化优先+BM25 混合回退 | 存着但未召回（`stored` vs `visible`）、兜底文本挤占 | `N3/N4`（全量 M）、`N7/N8/N11/N12`（BM25）、`N9/N10/N14`（实体属性匹配）、`N13`（事实优先+BM25 兜底） |

这些是候选方法，不是要求首轮全部实现。检索不取代 maintenance 主线；只有要解释读取/恢复影响时，才增加对应消融。每个 strategy_spec 保存各维度、参数、提示词与实现 hash；一组预设只是一个具体组合。

禁止把未来问题、答案或证据标签用于摄入阶段的重要性选择。依赖最终问题的读取只在最终问答阶段发生，并作为明确的读取策略记录。来源原文是否可再次读取是独立能力，不能隐藏在“压缩算法”内部。

### 2. agent 原生记忆能力清单

每个 agent/version/入口先审计以下能力；记录 configured_state、effective_state、可控性、观测程度与验证证据。

- 上下文压缩：自动阈值、轮前/轮中/轮后整理、兜底截断。
- 持久记忆：写入选择、合并、容量限制、删除、写入失败处理。
- 提示词注入：会话快照、按轮刷新、预算分配、优先级。
- 历史恢复：session 搜索、原文回读、外部 memory provider。
- 生命周期：session reset、轮末 flush、压缩前保存、重启恢复。
- 其他文本来源：工具结果裁剪、项目上下文、技能/用户画像中的事实。

effective_state 取值为 on/off/unknown；supported/unsupported 单列在 controllability；“不支持”与“已关闭”不是同一件事。未触发不等于已禁用；用足够长的输入或针对性样例验证。关闭 memory 工具也不等于关闭了注入、历史搜索和上下文压缩。

仅审计与实际入口有关的路径；某入口没有 gateway 时无需虚构 gateway 事件。与问题相关的原生机制应作为实验因子，不能一律关闭后就认为代表真实 agent。

### 3. 每种策略声明三种运行模式的适用性

| 模式 | 外加策略 | 原生机制 | 回答的问题 |
| --- | --- | --- | --- |
| native_only | 无 | 固定的原生 profile | 原 agent 在本协议下的基线 |
| replacement | 有，接管明确的目标能力 | 被接管能力经验证关闭，其余固定 | 替换这一维护能力是否有效 |
| coexist | 有 | 与 native_only 同一 profile | 在真实机制基础上叠加是否仍有效 |

另设 agent 内 no_external + target_off 作为同框架消融基线。direct 无 harness 只作独立参照，不能代替这个基线。

replacement 必须写清 scope，例如仅替换 context_compaction，不意味着持久记忆和历史恢复都已关闭。全部原生记忆关闭的 profile 可以作为额外条件，但不能混同“仅关闭目标能力”。

### 4. 最小可解释的 2×2 对照

选定一个外加策略 S 和一个原生机制 N，在同一个 agent 中固定其他条件：

| 单元 | 外加策略 | 目标原生机制 |
| --- | --- | --- |
| A | 无 | 关闭 |
| B | S | 关闭 |
| C | 无 | 开启 |
| D | S | 开启 |

对越高越好的同一指标 Y，策略在原生关闭时的增益为 B−A，在原生开启时为 D−C；交互项 I=(D−C)−(B−A)。对成本、错误率分别计算并说明方向，不把多个指标直接混成一个“增益”。

同一 question_id（共享历史时按 history_group_id 聚类）、模型、预算、数据版本、编排 profile 和会话策略配对，按题重采样给区间；总资源不同还须附成本指标。I 为正只能说明该配置下的正向交互估计，不能单凭聚合分数断言具体恢复机制。证据需来自事件轨迹。

先以“原生上下文压缩”为 N，再分别测试持久记忆写入、历史恢复、注入方式。不要一次关闭所有能力再把差异归因于其中某一项。某个单元无法实现或快照不可见时标 unsupported/partial；四格不齐就不报告完整交互项。

多个原生能力组合时先做单因素与少量预注册组合，不做完整笛卡尔积。先筛选代表策略，再对出现交互的组合做细化消融，控制实验量。

### 5. 记录实际执行链，而不只记录策略名称

每次维护事件记录 owner（external/native）、mechanism、phase、trigger、parent_event_ids、输入/输出快照及 hash、token、辅助调用和被影响的来源。parent_event_ids 为列表，支持合并多个输入。

例如：历史 → external 抽取 → native 摘要 → 预算裁剪 → 最终请求。实际顺序以日志为准，不能从配置猜测。明确外加策略输出进入哪个槽位，以及原生维护能否再次处理该槽位。

需要追踪的交互包括：

- 双重压缩：同一信息先后被两层改写，损失可能在第二层出现。
- 重复注入：同一信息来自原生 M 和外加 M，挤占上下文或产生冲突。
- 更新竞争：两层维护的版本不同，注入顺序影响新旧值使用。
- 恢复与掩盖：策略删掉证据，但原生历史搜索找回，因此最终正确不能证明策略无损。
- 写入未激活：固定历史导入未经过 agent 的自然写入路径，测到的是注入/读取而不是原生维护。
- 预算挤占：工具定义、原生摘要或用户画像减少外加策略可用空间。

去重率、来源追踪和版本冲突可由日志/标注计算；语义损失仍需文本证据。恢复次数只有观察到恢复事件才计数；没有观测能力记 null。

### 6. LongMemEval 固定历史如何触发原生维护

不能只把所有消息导入后问一次，就声称测过原生逐轮记忆管理。对每种 agent 声明 ingestion_activation：

1. native_lifecycle：按选定版本的实际维护生命周期触发，但保留数据集原始 user/assistant 消息。
2. explicit_maintenance_hook：固定历史摄入后显式调用维护 hook，记录调用与自然行为的区别。
3. context_import_only：只导入上下文，不能据此评价自然写入选择。
4. wrapped_history：用带角色与时间的文本包装摄入，包装提示词和维护调用计入开销。

在小样例中确认原生写入和压缩确实发生；不能触发则标“原生维护未激活”，不将零事件解释为无影响。因适配而产生的新 assistant 文本与原始历史分开，不能替换数据集消息。不同激活方式分组报告，不能混称原生表现。

### 7. 实验推进方式

第一阶段保留 N0–N4 作为 runner 的 smoke 预设，随后从新增维度选代表性策略，例如结构化版本记忆、混合摘要+原文、重要性淘汰。具体组合在查看评测结果前写入 strategy registry。

第二阶段完成 Hermes 能力审计，选择一个可控原生机制与一个策略组成 A/B/C/D。每个单元先用 5 题验证实际事件，再扩展固定 S 子集。后续策略复用相同 A/C 基线时须保持配置与版本一致并记录复用关系。

第三阶段对表现出交互的组合增加原生记忆/历史恢复消融，再扩展到其他 agent。实验单元数由明确 cell manifest 计算，不继续把所有计划都算成“5 种策略 × 30 题”。

### v0.2 四格适用范围

四格的策略开关还需匹配 identity 编排控制；不支持的模式声明 unsupported，不要求实现全部模式。目标机制关闭导致 overflow 时仅分析端到端 reliability 交互；质量交互使用按输入长度预选的容量可行子集，不能按运行成功与否筛题。详见[research-protocol.md](research-protocol.md)。

外部编排有独立 orchestration_profile_id，控制调度、hook、注入和状态访问；它与策略/native profile 分离。编排变化本身通过 identity 对照或相同策略配对研究，见[agent-integration.md](agent-integration.md)。

## 编排与扩展接口

采用简单文件和轻量适配接口即可；本文件规定边界，不要求建设插件平台。

### 独立扩展对象

| 对象 | 输入输出与责任 | 新增时的验收 |
| --- | --- | --- |
| dataset/scenario | 输入消息、时间与 source ID；评估标注旁路保存 | 时间顺序、角色、同题/同历史分组、标注不泄漏 |
| strategy | 可见状态 + 新摄入片段 → 新状态 + 事件 | 读取权限、预算、来源、确定性/随机性声明 |
| agent adapter | 固定历史摄入、状态快照、最终请求与原生事件 | 能力审计与实际激活证据 |
| orchestration | 调度、hook、注入槽位、状态路由、重试和会话边界 | identity 对照与调用顺序可追踪 |
| evaluator | 答案/证据 + gold → 标签与依据 | 固定版本、unknown/null 规则、人工抽检 |
| reporter | 只读结果与 manifest → 表格/图 | 不重新判分、不修改实验状态 |

### 编排 profile 最少字段

`orchestration_profile_id`、integration surface（direct runner/context engine hook/tool/file/wrapped history）、hook phase、执行顺序、注入 role/slot/position、是否可再被 native 处理、历史访问权限、session/reset 规则、snapshot/copy 方式、额外 prompt、调用/重试限制、实现 hash。

编排不是策略：相同摘要算法通过 tool 写入、轮前注入或 context hook 执行，可能有不同可见性和时机。profile 改变必须新建实验 cell；性能比较固定 profile，专门比较编排时固定策略与 native profile。

无外加策略时也经过相同编排路径的 identity/pass-through 控制，保留匹配的包装与 hook（不做无用 LLM 调用来假造等成本）。对比原生无 wrapper 入口与 identity wrapper 可估计包装影响。若无法实现等价 identity，明确缺失该控制，相关结果降级为系统级比较。

外加阶段的输出在何处终止必须明确：替换历史、追加 system memory、写文件或返回 tool result 是不同干预。原生 profile 记录其是否会重压缩、重注入或搜索恢复该输出。将 context_compaction、persistent_write、persistent_injection、history_recovery 分开配置，不能一个 memory_enabled 代替。

### 数据与会话契约

dataset adapter 先将 session ID/date/content 三者作为一个整体对齐，验证时间单调；排序使用稳定规则并保存原始顺序与转换 hash。Oracle 输入顺序尤其要验证，不能单独排序 dates 而错配内容。原始消息不改写、不翻译；时间解析不确定时保存原文并标记。

scenario manifest 声明语言、事实变化类型、干扰比例、目标长度、session 数、生成 seed/版本、标注来源、可评估前缀和开发/测试用途。配置样板见 [scenario-manifest.json](../templates/scenario-manifest.json)。同场景难度扩展形成新版本；不要根据某方法失败位置特制测试后宣称一般优势。

continuous：数据的 session 边界只是元数据，runner 工作上下文不重置；该组不能充分代表自然跨会话生命周期。session_reset：先执行指定的原生/外部 flush，再结束 session，清空工作历史，按 profile 重载持久状态。明确 external M 是否持久、磁盘历史是否可搜索、最终问题是否开启新 session。不得偷偷把完整旧 H 注回新会话。

对照相同生命周期；原生真实性要求不同时另建系统比较组。如果 agent 不能在保留固定历史的同时自然激活写入，就使用 explicit hook 诊断并标注适配，不能虚称 native_lifecycle。

### 快照与探针契约

复制必须包含所有影响下一步的状态：原生/外部 M、工作历史、session 数据、检索索引、工具状态、计数器与随机状态（若可得）。远程共享 memory provider 不能隔离时，该 trace 模式标 unsupported。

保存 state hash 与 branch_id；probe 的调用、写入和恢复只在分支发生。维护前后比较分为两种：snapshot_readout 用固定 reader 读取快照、禁止维护/恢复，用于诊断；operational_probe 在复制分支中照常执行 agent，用于观察可用能力。两种成本和准确率分开，前者不称原生表现。否则前置探针触发同一次压缩会抹去“维护前”的意义。

探针前后保留 question/answer version 与 eligibility；重放失效区间优先读取已存快照，重新生成状态只能视为复现实验，不能覆盖原轨迹。

### 新方法或新 agent 的最小交付

只需新增一份 spec/profile、必要 adapter 实现、能力证据、comparison/cell 条目和小样例验收。不得复制一整套评估和展示代码。未知字段保留 null，不用默认“支持”填充。

实现完成后将状态由 proposed/design_only → implemented → smoke_verified → frozen；只有 frozen 配置可进入正式比较。不同阶段不能只靠存在一个 JSON 文件推断。

## Hermes 接入

此处将 Hermes 解释为 Nous Research 的 Hermes Agent。资料核查日期：2026-11-16。尚未下载、安装或固定本地 commit；以下是待落地的实验协议，具体 hook、配置键与启动方式需以所选 commit 验证。

### 已核实、与实验直接相关的行为

Hermes 使用 `MEMORY.md` 与 `USER.md` 保存持久记忆，通过 memory 工具维护。官方文档描述其会话起始注入为冻结快照，因此文件中途写入不等于本轮系统提示已经更新；跨会话实验必须显式记录会话边界。[官方持久记忆文档](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory)

官方上下文文档描述可替换的 ContextEngine，以及 agent 内压缩与 gateway 会话整理两层机制。控制实验必须检查所有实际启用的维护入口，不能只修改一个阈值就宣称关闭压缩。[官方上下文压缩文档](https://hermes-agent.nousresearch.com/docs/developer-guide/context-compression-and-caching)

来源仓库：[NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent)。上面的行为是文档核查结果，不是本地集成测试结果；不依赖可变化的默认容量与阈值。

### 最小接入原则

优先选择 CLI 或官方程序接口的单进程文本入口；首版不接聊天 gateway。使用固定文本回放，禁用与本实验无关的网络、执行工具、技能学习和外部记忆 provider；H0 保留被研究的原生记忆机制并记录限制后的有效工具集。

每个 run 使用独立 Hermes home/profile 与 session 数据。一个 run 的跨会话测试复用该 run 的持久状态；不同策略和重复实验从相同干净初始状态开始。探针使用状态副本，探针写入、压缩、session_search 记录不得污染主 run。

H1/H2 需确认历史搜索、持久记忆读写和隐含恢复路径的实际关闭效果。配置中写了 false 不算证据，要检查实际消息和工具事件。无法采集最终模型请求时，prompt_visibility 标记 partial，不能据此得出“信息明明可见却没用对”的结论。

### 适配步骤与验收

| 步骤 | 交付物 | 验收 |
| --- | --- | --- |
| 固定版本 | commit SHA、依赖锁定、入口与生效配置 | 相同环境能重复启动；不使用浮动 main 作为版本 |
| 单轮回放 | 输入/响应和 request_id 映射 | 用户轮与内部步骤可对齐 |
| 观测原生压缩 | 触发原因、前后快照、辅助调用 | 小预算对话至少触发一次，核对是否还有第二条压缩路径 |
| 观测持久记忆 | 写入事件、磁盘快照、注入快照 | 同 session 与新 session 的可见性可区分 |
| 完成 H1/H2 | 禁用路径的执行证据 | 无隐式历史搜索或 provider 恢复 |
| 替换压缩 H3/H4 | 与 N3/N4 相同的策略输入输出协议 | 所有维护事件归属明确；原生兜底触发则标为混合策略 |
| 生成统一日志 | 与 N 系列同格式 | 能使用同一评估和展示脚本 |

若选定版本有稳定的 context engine 扩展点，优先使用；否则做局部适配并保存 patch 与 hash。不要为首批实验重构 Hermes。如果无法完整接管压缩，先交付 H0/H1/H2 观测实验，H3/H4 保持 blocked 状态并说明原因。

H 与 N 的主模型可以相同，但系统提示、工具定义与调度存在额外开销。报告请求总预算、正文预算、辅助调用以及 wall time，跨 harness 比较只作为系统级结果，维护策略的因果解释优先使用同 harness 内对照。

### 原生机制也是实验变量

遵循[agent-integration.md](agent-integration.md)，先输出版本/入口级 capability audit，再构建 native profile。H0–H4 仅保留为起步预设，不替代逐策略的原生机制开/关对照。每种策略须注明 replacement scope 或 coexist 顺序，并观测重复压缩、重复注入、版本冲突与历史恢复。

LongMemEval 的固定历史摄入必须记录 ingestion_activation。若导入绕过自然写入路径，只能报告 context_import_only 的结果，不称为 Hermes 自然维护基线。实际原生维护是否触发需有事件证据；无法关闭的能力不能标 off。

### 编排与可行性验收

使用 [agent-capability-audit.json](../templates/agent-capability-audit.json) 和 [orchestration-registry.json](../configs/orchestration-registry.json)；后者仅定义待实现 profile，不宣称 Hermes 接口已支持。先验证 identity 包装、固定历史下维护激活、状态副本及目标能力关闭，再冻结四格单元。不满足条件时保留 observational 结果；能力与编排的标准见[agent-integration.md](agent-integration.md)。

### 触发暴露与调度匹配

native_enabled 仅说明能力可用，另记录 trigger_opportunity_count 和 executed_event_count。策略在上游减小输入，可能令下游 native 阈值永远不触发；这是共存配置下的调度交互，不是已证明两次压缩均发生。需要研究二次处理时另设固定维护时点实验，并标 explicit_maintenance_hook，不能混入原生自然调度成绩。

质量四格需要同时满足容量可行与预先安排的维护触发机会。外部与原生机制是否实际作用，由事件链说明；不能只看 on/off 配置或只筛选已触发题目。
