# DeutschTune

**德语大模型微调、评测与远程 GPU 推理部署工程**

**DeutschTune — German LLM Fine-Tuning, Evaluation & GPU Serving**

> 仓库简介：*Qwen2.5 LoRA fine-tuning, frozen-test evaluation, and local Flask/vLLM serving for German word restoration.*

## Project Overview

DeutschTune is an engineering case study for a **closed-form German word-restoration task**. It covers data validation, Qwen2.5-7B-Instruct LoRA fine-tuning, paired Base/LoRA evaluation on a fixed test split, and a vLLM-to-Flask serving path. This repository contains selected source and aggregate evidence, not the corpus, model, adapter, raw predictions, credentials, or a live deployment.

## Problem

输入为带缺词位置、选项或字母线索的德语句子；目标是恢复指定词语并遵守输出格式。任务范围是受约束的文本 SFT，不代表自由德语教学、开放问答或通用德语能力。

## Architecture

```text
Tatoeba 来源记录 → D01 数据校验与切分 → T01 Qwen2.5 LoRA
                                 ↘ E01 冻结 Test 的 Base/LoRA 离线 Silver 评分
模型/Adapter → I01 vLLM OpenAI 兼容接口 → A01 Flask 本地 API → 外部系统单题联调（历史记录）
```

仓库收录 D01 校验与训练/验证准备代码、E01 评分器、I01 客户端和 A01 本地 Flask 接口。原始数据生成与批次准入实现、远端部署包及外部系统适配源码不在此公开副本中。模块关系见 [架构](docs/ARCHITECTURE.md)。

## Implementation

- **技术栈：**Python、PyTorch、Qwen2.5-7B-Instruct、LoRA、LLaMA-Factory、vLLM、Flask；历史训练运行于 Linux A800 GPU，工程交接使用 SSH。精确训练 PyTorch 版本未由归档单独证实。
- **LoRA 选择：**冻结基座并训练可独立保存的低秩适配器，以较小的可训练参数量完成该受约束任务；项目没有进行全参数微调对照。
- **数据：**历史实验划分为 7,500 train / 1,000 validation / 1,000 独立 test；公开仓库不含句子、逐句来源表或参考答案。[数据边界](docs/DATA_PIPELINE.md)
- **训练：**Qwen2.5-7B-Instruct、LLaMA-Factory 0.9.5、BF16 LoRA；历史记录为 2 epochs、1,876 optimization steps，train loss 0.0196、eval loss 0.0011、退出码 0。[训练配置与证据等级](docs/TRAINING.md)
- **评测：**同一冻结 Test 的 Base/LoRA 配对推理；参考答案只进入离线评分。评分为机器 Silver，未完成人工 Gold。[评分口径](docs/EVALUATION.md)
- **服务：**历史 GPU 使用 vLLM 0.31.0 与 Flask/Gunicorn。公开代码提供 loopback 限定的 `/tatoeba/restore` 本地示例；外部系统单题联调有历史证据，但平台适配源码因公开权利尚未确认而排除。[服务边界](docs/SERVING.md)

## Evaluation

| 冻结 1,000 条 Test 的机器 Silver 指标 | Base | LoRA |
|---|---:|---:|
| 选词匹配 | 912/1,000 | 1,000/1,000 |
| 原句严格匹配 | 893/1,000 | 1,000/1,000 |
| 规定格式合规 | 0/1,000 | 1,000/1,000 |

这些是历史本地冻结档案中的固定任务结果，本次公开副本没有重新推理或重算，也不能外推为通用德语准确率。原始预测与 Test 参考均未公开。[结果摘要](results/G5_PUBLIC_SUMMARY.md)

## Deployment and local API

历史服务链为 vLLM（GPU 回环地址）→ Flask/Gunicorn → 受控外部接口。归档日志支持历史模型服务路由 HTTP 200；不证明当前 GPU 在线。公开版只保留本地接口示例，默认并强制绑定 loopback，入站请求体默认限 256 KiB。该接口没有入站鉴权，**不要映射到公网**。

在自行准备并获准使用 OpenAI 兼容模型端点后，可从仓库根目录运行：

```powershell
python -m pip install -r requirements-api.txt
python -m api.server
```

默认配置见 [serving.example.json](configs/serving.example.json)，健康检查为 `GET http://127.0.0.1:20995/healthz`，任务接口为 `POST /tatoeba/restore`，请求字段为 `request_id` 与 `text`。此启动命令是候选示例说明，**未在 R3 对真实模型端点执行**。历史外部系统单题演示不等于生产部署。

## Evidence and reproduction

公开副本提供[聚合证据索引](results/EVIDENCE_INDEX.md)、[脱敏训练 YAML 示例](configs/lora_tatoeba.example.yaml)和本地合成测试。YAML 中的 `A4_*` 环境变量及 `a4_*` 数据集键是历史接口兼容标识，安全改名需同时迁移数据注册与代码调用，因此保留；它们不代表对外项目名称。

本地代码测试命令：

```powershell
python -m pip install -r requirements-dev.txt -r requirements-api.txt
python -m pytest -q -p no:cacheprovider
```

测试仅覆盖合成输入与代码边界。完整训练和冻结评测需要另行获准的数据、模型、算力和私有证据；公开副本尚未做端到端重训或部署复现。[复现说明](docs/REPRODUCTION.md)

## Repository map

- `api/`：本地 Flask 服务、响应适配和既有 Tutor 门禁兼容模块。
- `scripts/`：数据契约、训练/验证准备、OpenAI 兼容客户端、离线评分。
- `configs/`：脱敏示例，不是原始 GPU 配置。
- `tests/`：运行时构造的合成用例与接口安全回归。
- `docs/`：架构、数据、训练、评测、服务、复现和第三方说明。
- `results/`：仅含聚合结果与证据边界。

## Limitations and rights

Tatoeba 来源文本及其 CC BY 2.0 FR/CC0 行级义务、Qwen 模型、LLaMA-Factory 和 Python 依赖分别受各自条款约束；项目根许可证不能覆盖第三方资产。公开副本无数据、权重、Adapter、原始日志和平台截图。翻译关联及语义泄漏未完全核实，人工 Gold 未完成，当前服务在线状态未验证。[限制](docs/LIMITATIONS.md) · [第三方说明](docs/THIRD_PARTY_NOTICES.md)

仓库根目录保留初始仓库提供的 Apache-2.0 `LICENSE`。该许可证仅适用于项目有权许可的内容，不延伸至 Tatoeba 数据、Qwen 模型、LLaMA-Factory、第三方依赖或课程材料；具体边界见[第三方说明](docs/THIRD_PARTY_NOTICES.md)。
