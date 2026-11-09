# 回答与拒答规则

本次修复采用严格拒答策略，不自动重新生成答案。

## 放行条件

以下条件必须同时满足：

1. 答案非空，并存在来源片段。
2. 每个非空文本句子都标注引用，引用编号属于本次提供的来源范围。
3. 引用报告确实对应当前答案和解析出的全部引用，并已执行校验。
4. 所有引用的判定均为 `True`。不支持、未配置校验、调用失败、空回复和无法解析的回复都不能放行。
5. 回答完整度达到最低要求。
6. 综合置信度达到总阈值。

当前按句子执行保守的引用检查，不判断句子是否属于事实陈述；没有引用的标题、引导语也可能触发拒答。
模型判断仍可能出错，这些检查不能保证答案绝对正确。

## 两条独立的分数线

- `RAG_MIN_COMPLETENESS`：API 默认读取的完整度最低分，初始值 `0.5`。
- `RAG_CONFIDENCE_THRESHOLD`：原有综合分数线，默认 `0.5`。
- 直接使用 `UnknownHandler` 时，通过 `min_completeness` 和 `threshold` 参数配置。

初始值不是经过真实数据校准的最佳值，后续应使用自己的问题集评估。
调整环境配置后，需要重启服务才能加载。此修复没有重启现有后台服务或改写持久化索引。

例如，检索相关度 `0.8`、引用支持比例 `0`、完整度 `0.9` 得到综合分 `0.59`。
即使总分高于 `0.5`，不支持的引用仍会独立触发拒答。

## 拒答原因在哪里

为了保留接口兼容性，原有 `declined.reason` 仍为 `insufficient_confidence`。
具体失败项位于：

```text
confidence.details.gate_failures
confidence.details.min_completeness
declined.summary
```

API 拒答时 `answer` 返回 `null`，不展示未通过检查的候选答案。

## 判定词修复

`UNSUPPORTED` 包含子串 `SUPPORTED`，因此不能使用子串包含判断。
现在只完整匹配 `SUPPORTED` 或 `UNSUPPORTED`（忽略大小写和首尾空白）。
兼容 JSON 中真正的布尔值 `supported: true/false`；其他格式视为未知。
未知保持 `None`，不与“支持”混淆。

完整度判分调用失败或返回非有限数值时按低分处理；没有可识别的关键词也不再自动获得满分。
未配置模型时的词法完整度只是近似检查，不代表真正理解了回答。

## 裁判提示与普通问答分离

引用与完整度检查现在直接发送专用系统提示和明确的 JSON 待评数据，不再附加普通问答的引用要求或“问题：评分任务”包装。

- 引用输入是 `claim` 和 `source`，输出为布尔 `supported` 及文字 `reason`。
- 完整度输入是原始 `question` 和现成 `answer`，输出为范围 `[0, 1]` 的数值 `score` 及文字 `reason`。
- 兼容完整 JSON 对象、单个 JSON 代码块，以及原有独立的 `SUPPORTED` / `UNSUPPORTED` 判定词。
- 不接受混杂解释的 JSON、重复字段、字符串布尔值、布尔型分数或越界分数；解析失败仍保持拒答。
- 待评答案和来源放在用户数据中，不再把原文拼进高优先级的系统指令。

校验故障与“内容确实不受支持”分开记录：

```text
confidence.details.citation_judge_errors
confidence.details.completeness_judge_error
```

`supported: false` 是内容判定，不计为协议故障。未知判定、格式错误及接口失败仍不能放行。本次裁判修复没有降低完整度或综合分数线，也没有放宽逐句引用规则。

固定 20 题的基线、裁判修复、中文重排对照分别保存于 `data/benchmarks/crud_rag_20/evaluation/`，避免覆盖原始失败证据。独立语义评分使用同一模型的另一套提示，分数仍存在模型评审偏差。

## 运行测试

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests -q -p no:cacheprovider
```

重点测试文件：`tests/test_answer_gates.py`、`tests/test_citation.py`、`tests/test_confidence.py`、`tests/test_api.py`。
这些测试使用隔离的模拟判分结果，验证控制逻辑，不调用真实模型，也不代表真实回答质量指标。
