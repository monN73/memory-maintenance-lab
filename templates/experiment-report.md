# 实验报告：<experiment_id>

状态：未运行 / 完成 / 部分完成。数据性质：真实运行 / 人工示例。

## 假设与设置

- 研究问题与预注册失效定义：
- dataset/revision/variant、subset ID 清单、question 数、protocol：
- evaluator commit、judge、ingestion mode、session policy：
- scenario/config/prompt hash：
- 模型、tokenizer、Hermes commit（如适用）：
- 输入预算、输出预算、记忆预算、工具权限与维护阈值：
- 重复次数、seed 支持情况、运行日期：

## 汇总

| 策略 | 完成/计划 run | 探针准确率 | 当前事实保留率 | 首错/删失 | overflow | 总 token / 成本 |
| --- | --- | --- | --- | --- | --- | --- |
| 待运行 | — | — | — | — | — | — |

最终成绩另列正确/计划、正确/已判定、预测覆盖率与 judge 覆盖率；按题型分组，abstention 单列。Oracle、S、M 与不同预算不合并。

## 逐轮失败证据

run_id、question_id、session_id、turn_id、fact_id（若已标注）；原文来源；维护前后文本；实际提交上下文；探针答案与判定。
区分存储丢失、注入缺失、过时信息、回答错误与无法归因。

## 有效性与下一步

列出缺失观测、未失效样本、运行错误、judge 抽检及控制变量偏差。
给出由证据支持的发现和下一轮单变量实验；不要将人工示例当作实验结论。

公开数据缺少前缀标注时，不报告精确遗忘轮次或事实保留率。最终错误与 maintenance_trace 诊断分别展示。

## 外加策略与原生机制交互

列出 strategy_spec、native profile、replacement scope、摄入激活方式和能力审计证据。按同 agent 的 A/B/C/D 四格报告成绩、覆盖率与成本；单元完整时计算策略条件增益和交互项。

提供 external→native→实际请求的事件链，区分二次压缩、重复注入、版本冲突和原生恢复。无法控制/观测的能力及配置失配 run 单列，不给出超出证据的因果结论。

## 比较与复现声明（v0.2）

comparison_id、claim_scope、changed_factors、orchestration_profile、identity 控制证据、history_group_id 与开发暴露；主要指标、预设样本、重复和运行顺序。分别报告算法、agent 内与系统比较。四格注明 quality/reliability，容量不可行时不解释为纯质量交互。

引用冻结的 run manifest；注明 target_unreachable、辅助模型预算、judge 缺失、竞争终点、抽样区间与人工抽检。结果尚不支持的判断明确列为后续假设。
