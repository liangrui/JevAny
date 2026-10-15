# JevAny 代码深度解读 · 三观递进写作方案

> 交付物：三篇长文（整体观 → 具体观 → 深刻不忘观），全部产出到 `/workspace/ReadCode/`。
> 方法论：`read-code` 技能的"三观递进"框架；写作风格：总分总 + 层层递进；语言：中文（术语首现附英文）；深度：论文级。

---

## 一、总述（Summary）

### 1.1 任务目标
对 `/workspace` 下的开源项目 **JevAny** 做一次"由表及里、由宏观到微观再到本质"的代码深度解读，产出 3 篇层层递进、各自独立可读、图文并茂（每篇 ≥3 张 Mermaid 图并配图注）、并内置"使用指南 + 案例"的中文长文。

### 1.2 交付物清单
| 文件 | 定位 | 一句话主旨 |
|---|---|---|
| `/workspace/ReadCode/01-整体观-项目全貌解读.md` | 看见（鸟瞰全貌） | JevAny 是什么样的基础设施、在生态中处于什么位置、代码如何组织、端到端怎么跑 |
| `/workspace/ReadCode/02-具体观-算法与实现剖析.md` | 看懂（深入细节） | 指针头 / 决策 token 打包 / SFT-软目标 / RLCR / 免训练 readout / 推理加速，逐点给出论文原理与代码例证 |
| `/workspace/ReadCode/03-深刻观-哲学与升华.md` | 看透（升华本质） | System 1 vs System 2 的分工哲学、委托选择的边界、核心主题升华，结尾以两三句话收束 |

> 用户要求"用三篇讲述即可"且"使用指南与案例融入三篇内"：**不额外产出第 4 个文件**，使用指南与案例作为每篇末尾的一节。

### 1.3 三篇的递进关系（认知升级路径）
```
看见（它是什么） ──► 看懂（它怎么做） ──► 看透（它为何如此）
  建立地图            填充机理            凝练洞见
01 回答"在哪、有何、怎么连"
02 回答"靠什么算法/数据/工程实现，论文依据是什么"
03 回答"这套设计背后的一以贯之的哲学是什么，永不忘记的两三句话是什么"
```

---

## 二、当前状态分析（Current State Analysis）

> 以下均来自 Phase 1 的实际探索（文件读取 + 三个并行检索代理），非假设。

### 2.1 项目画像
- **名称/版本**：`jevany` `0.3.0`，作者 Tianxin Wei；Apache-2.0；派生自 [Kev](https://github.com/jaredpalmer/kev)（见 `NOTICE`、[model.py](file:///workspace/jevany/model.py#L1-L2) 头注释）。
- **自我定位**（[README.md](file:///workspace/README.md#L25-L29)）：*"Open infra for System 1 decision model training and deployment"* —— 覆盖数据准备、模型适配、评估的一体化基础设施；标语 *"Your Jev from Any Model to Any Application"*。
- **核心命题**：不做文本生成，而是"给一个共享 state + 若干带候选的定型问题，直接返回**选择**及**概率**"。三个问题类型：`choice` / `noul`（yes-no）/ `score`（有序等级）（[API.md](file:///workspace/docs/API.md)）。
- **语言/依赖**：Python ≥3.12，核心依赖仅 `pydantic`；`torch/transformers/peft` 等在可选 extra（`local/serve/train/multimodal/bedrock/demo/robotics/fast`）中（[pyproject.toml](file:///workspace/pyproject.toml)）。CLI 入口 `jevany.cli:main`。

### 2.2 代码结构（目录 → 职责）
- `jevany/`（核心库，~45 个模块）：`model.py`（决策 token 打包、指针头、分支掩码、`DecisionModel`）、`backbones.py`（决策 token 定义与适配器）、`train.py`（损失函数与训练循环）、`training.py`（TOML 配方解析 + Python API）、`inference.py`/`runtime.py`/`serve.py`/`api.py`/`client.py`/`openai_compat.py`（推理与服务）、`cudagraphs.py`/`fused_kernels.py`（性能工程）、`readout.py`/`predictors.py`/`letter_predictor.py`/`letter_readout.py`/`letter_runtime.py`（读出头）、`data.py`/`datasets/*`（数据构建）、`agent_harness.py`/`harness.py`/`symbolic.py`/`terminal_harness.py`/`webshop_harness.py`/`webarena_harness.py`/`bedrock.py`（Agent 委托与符号推理）、`benchmark.py`/`suite.py`/`metrics.py`/`external_eval.py`（评估）、`placement.py`/`checkpoint.py`/`resume.py`/`device.py`/`self_train.py`。
- `jevany/datasets/`：`build_sft.py`、`build_rlcr.py`、`convert.py`、`starter/`（合成工单样例）。
- `jevany/demos/`：`server.py`、`games.py`、`arm.py`、`arm_policy.py`、`context.py`、`static/`、`recordings/`（playground）。
- `docs/`：`ALGORITHM.md`、`API.md`、`DATA.md`、`TRAINING.md`、`DEPLOYMENT.md`、`EFFICIENCY.md`、`EVALUATION.md`、`EXTERNAL_EVALUATION.md`、`CHOICE_READOUT.md`、`LETTER_READOUT.md`、`INTEGRATIONS.md`、`PLAYGROUND.md`、`RESEARCH.md`、`ROBOTICS.md`、`CASES.md`、`experiments/AGENT_HARNESS_FRONTIER_PROTOCOL.md` 等。
- `recipes/`（`sft.toml`/`rlcr.toml`/`finetune.toml`）、`results/`（机器可读结果 JSON/SVG）、`scripts/`（~40 个构建/评估/绘图脚本）、`tests/`（~55 个测试，含 `test_model_support.py`、`test_fused_kernels.py`、`test_cudagraphs.py`、`test_agent_harness.py`）、`site/`（静态站点 + 文档 HTML）、`reports/`（技术报告 PDF）。
- `examples/`：`request.json`、`sql_repair.py`、`tool_routing.py`、`service_recovery.py`、`inbox.py`、`bedrock_*.py` 等端到端用例。

### 2.3 已确认的核心技术事实（第 2 篇的"事实基座"）
1. **决策 token 体系**：5 个分隔符 `<|jev_state|> <|jev_question|> <|jev_option|> <|jev_end_option|> <|jev_decide|>`，优先复用主干中"罕用的现存特殊 token"，**不新增 embedding 行**，由 LoRA 赋予其语义（[model.py](file:///workspace/jevany/model.py#L12-L18)、[backbones.py](file:///workspace/jevany/backbones.py)）。
2. **打包布局**（[model.py `encode`](file:///workspace/jevany/model.py#L97-L135)）：`<state> state | <q> instr <opt> o </opt>… <decide>`；产出 `ids/seg/pos/opt/decide_idx/opt_idx`。`MAX_STATE=1024, MAX_BRANCH=2048, MAX_PACKED=2048`。`option_isolation=True` 时每个 option 为独立子分支 → **置换不变性**。
3. **两类分支模式**（[model.py](file:///workspace/jevany/model.py#L212-L260)）：`packed`（block-causal 掩码）/ `rows`（每题独立因果行、重复 state），后者用于循环/滑窗类主干（Qwen3.5/3.8 的 Gated DeltaNet）。
4. **指针头**（[PointerHead](file:///workspace/jevany/model.py#L263-L286)）：`q=Q(h_decide), k_i=K(h_i)`，`z_i = k_i·q/√d_p + R(k_i*q)`，`d_p=256`，`R` 为 256 宽 `Linear/GELU/Linear` 且**输出层零初始化**的残差打分器；`temperature` 仅推理期除 logits，**不改变 argmax**。
5. **direct-token（lm_token）读头**：给候选分配固定单 token 标签（`A–Z,a–z,0–9` + Unicode 扫描，`MAX_DECISION_OPTIONS=255`），在 `<decide>` 处**取冻结 LM 头对应 logits 并在候选上归一化**（[model.py `decision_verbalizers`](file:///workspace/jevany/model.py#L39-L75)）。
6. **损失函数**（[train.py](file:///workspace/jevany/train.py#L28-L85)）：SFT 硬标签 `-log p_y` 或软目标 `-Σ t_i log p_i`；`lm_token_loss` 用**全词表分母**、推理时再在候选上重归一；RLCR 用 `r(c,q)=c-(q-c)²`（Damani et al.）+ 32 个零均值高斯扰动 → 组内相对优势 → 各向同性高斯位置分数（**按 option 维度求和而非求均值**）。配方：σ 0.4→0.1、policy_weight 0.25、ce_weight 0.5、梯度裁剪范数 1。
7. **免训练 choice-token readout**：将候选标注为 `A–Z,a–z`（≤52），取答案位 logits 重归一；**可改变被选答案**（与温度缩放不同），可在 dev 上拟合一个混合权重（[README.md](file:///workspace/README.md#L285-L319)）。
8. **推理加速**（[fused_kernels.py](file:///workspace/jevany/fused_kernels.py)、[cudagraphs.py](file:///workspace/jevany/cudagraphs.py)、[EFFICIENCY.md](file:///workspace/docs/EFFICIENCY.md)）：FLA 融合 kernel（RMSNorm/SwiGLU/Gated DeltaNet/Attention 多 GEMM 合一）+ fused SDPA + **按 padded length 分桶的 CUDA Graph**（`LENGTHS=(128…16384)`）。实测：4B/A100 4.1×、27B/H200 3.72×、30B/H200 2.33×。
9. **服务与契约**（[serve.py](file:///workspace/jevany/serve.py)、[API.md](file:///workspace/docs/API.md)）：`POST /v1/systemone`、`GET /v1/models`；置信度公式 `choice: (max_p−1/K)/(1−1/K)`、`score: 1−E[|level−mode|]/(L−1)`；与 TypeSafe Jev API 契约对齐。
10. **Agent 委托**（[agent_harness.py](file:///workspace/jevany/agent_harness.py)、[AGENT_HARNESS_FRONTIER_PROTOCOL.md](file:///workspace/docs/experiments/AGENT_HARNESS_FRONTIER_PROTOCOL.md)）：`BedrockJevAgent` 三种模式（`baseline/optional/jev_only`）+ `delegate_to_jev` 工具；**D0–D4 委托阶梯**与 **T0–T4 任务层级**两轴独立；黄金法则"委托选择瓶颈，而非委托任务"。
11. **评估体系**（[EVALUATION.md](file:///workspace/docs/EVALUATION.md)）：Transfer-v9（1046 决策）/ JevBench（231 公开开发项）/ Typed Decisions / JevJudge；指标 accuracy / NLL / Brier / ECE；五个 LoRA 发布模型合计约 2,082 H200 GPU-hours，训练数据 1,772,725 条记录 / 2,180,242 个标注决策。

### 2.4 设计理念（第 1、3 篇的主线）
- 把"选择"从"推理"中剥离：LLM 负责规划/生成候选/恢复/验证，Jev 只在**有限候选集**上做**一次性前向选择**并返回**可校准的概率**。
- 以 `state + questions` 的统一契约贯通训练与推理（"训练所见即推理所用"）。
- 温度校准与 argmax 解耦（"改变置信度，不改变答案"）——强调**可校准性（calibration）**是第一等公民。
- 工程上追求单 batch、低延迟、可复现的"极致工程"。

---

## 三、拟变更 / 拟产出（Proposed Changes）

> 本章逐文件给出"写什么、为什么写、怎么写"，达到**无需二次决策即可执行**的粒度。

### 3.0 通用写作规范（三篇共同遵守）
- **结构**：每篇严格"总述 → 分述 → 总结"三段式；节内继续层层递进（宏观 → 中层 → 细节）。
- **图注**：每张 Mermaid 图**上方或下方**用一句话说明"这张图展示了什么、为什么重要"。
- **代码引用**：使用 `file:///workspace/...` 绝对路径可点击链接（遵循 Code Reference 规范），行号区间标注。
- **术语**：首现给出 `中文（English）`；公式使用 Markdown 代码块（不放进 Mermaid 节点，避免语法风险）。
- **诚实性**：所有数字、结论必须来自仓库内实际文档/代码，不臆造；引用论文必须真实存在且链接有效。
- **每篇末尾**统一设 `使用指南与案例` 一节（见 3.4）。

### 3.1 文件一：`/workspace/ReadCode/01-整体观-项目全貌解读.md`

**总述（开篇 2–3 段）**：JevAny 是什么、解决什么问题、核心价值主张（System 1 决策基础设施）。

**分述（五层递进）**：
1. **生态定位**：在"决策模型"生态中的位置——对比 Kev（jaredpalmer/kev）、TypeSafe Jev、Laya、decider、Intern-Decision 等基线与本项目的关系（数据取自 [EVALUATION.md](file:///workspace/docs/EVALUATION.md)、[EFFICIENCY.md](file:///workspace/docs/EFFICIENCY.md)）。说明"从任意主干模型到任意应用"的适配能力（26 个受支持模型族，[supported-model-families](file:///workspace/docs/supported-model-families.svg)）。
2. **整体代码结构**：目录树 + 模块职责表；把 ~45 个核心模块按"数据 / 模型 / 训练 / 推理服务 / 读出头 / Agent 委托 / 评估 / 性能工程"归纳为 8 大层。
3. **设计理念**：state+questions 统一契约、候选约束而非自回归生成、概率可校准、委托选择而非委托任务。
4. **实现原理概述**：5 决策 token + 指针头/direct-token 两条读头 + LoRA 适配 + 校验（数据准入/上下文长度/JSON schema）。
5. **整体流程**：三条端到端主线——(a) 训练链路（`data init/validate → train(SFT→RLCR) → checkpoint`）；(b) 推理/服务链路（`serve → POST /v1/systemone → runtime.answer → 概率 → 响应`）；(c) Agent 委托链路（LLM 生成候选 → `delegate_to_jev` → Jev 选择 → 回传/回退）。

**配图（≥3，实际给 5 张，均配图注）**：
- 图 1：**生态定位图**（`flowchart`）——主干模型族 / JevAny 基础设施 / 应用与基准的相对位置。
- 图 2：**分层架构图**（`flowchart LR` + `subgraph`）——8 大层与其依赖方向。
- 图 3：**目录结构树图**（`flowchart`/`graph TD`）——顶层目录 → 关键子模块。
- 图 4：**端到端流程图**（`flowchart TD`）——训练与推理两条主线并行展示。
- 图 5：**决策 token 打包布局图**（`flowchart`）——state/branch/option/decide 的序列与 `seg/pos/opt` 三元索引。

**总结（收束段）**：一段话概括"整体画像"——一个把决策从推理中剥离、以统一契约贯通训推、以极致工程求低延迟的基础设施。

### 3.2 文件二：`/workspace/ReadCode/02-具体观-算法与实现剖析.md`

**总述**：点出"最值得深挖的 6 个技术点"，并给出它们如何共同支撑项目目标。

**分述（每个技术点：原理/论文 → 代码实现 → 具象例子）**，共 6 节：
1. **决策重构：从"生成答案串"到"候选打分"**
   - 代码：`encode()`、`decision_tokens()`、`user_tokens()/safe_text()`（防伪分隔符）、`branch_mask_batch()`、`option_isolation` 的置换不变性。
   - 例子：用 `examples/request.json` 走一遍 token 打包，展示 `ids/seg/pos/opt` 与 `decide_idx/opt_idx`。
   - 论文：序列打包/块因果掩码相关（见 3.3 检索清单）。
2. **指针头（PointerHead）与残差打分**
   - 公式与零初始化残差层；`temperature` 校准只改置信度不改 argmax。
   - 论文：注意力打分（Scaled Dot-Product，Vaswani 2017）；温度标定（Guo 2017）。
3. **direct-token 读头与 verbalizer 选择**
   - `decision_verbalizers()` 的"单 token 往返"约束；全词表分母训练 vs 候选集归一推理的差异与动机。
   - 论文：LM head 词表约束 / constrained decoding 类工作。
4. **SFT：硬标签 CE 与软目标（ambiguity as a distribution）**
   - `question_loss` 的硬/软两分支；软目标把"信息缺失/歧义"编码为分布。
   - 论文：标签平滑/蒸馏（Hinton 2015）、proper scoring rules（Gneiting & Raftery 2007）。
5. **RLCR：校准奖励的组相对策略梯度**
   - `rlcr_reward = c-(q-c)²`、32 组高斯扰动、组内基线、各向同性高斯位置分数（**对 option 维求和**的抗退化设计）。
   - 论文：Damani et al.《Beyond Binary Rewards》；Brier score；REINFORCE 基线。
   - **诚实结论**：SFT 为默认发布；RLCR 在开发集 NLL −0.005、迁移精度未提升（负结果，按仓库文档如实陈述）。
6. **推理加速：融合 kernel + CUDA Graph**
   - FLA 融合（RMSNorm/SwiGLU/Gated DeltaNet/Attention）、fused SDPA、分桶 CUDA Graph、BF16 LoRA 合并、`torch.compile` 的适用边界与取舍。
   - 论文：FlashAttention（Dao 2022）、Gated DeltaNet（Yang 2024）、Mamba（Gu & Dao 2023）、SwiGLU（Shazeer 2020）、RMSNorm（Zhang & Sennrich 2019）。
   - 数据：引用 [EFFICIENCY.md](file:///workspace/docs/EFFICIENCY.md) 的实测加速比与精度对齐。

**配图（≥3，实际给 6 张，均配图注）**：
- 图 1：指针头计算图（`flowchart LR`）。
- 图 2：`encode()` 打包 + packed/rows 两种掩码对照（`flowchart`）。
- 图 3：RLCR 组相对策略梯度流程（`flowchart TD`）。
- 图 4：推理加速阶梯（`flowchart LR`：Default → +kernels → +fused SDPA → +CUDA graphs）。
- 图 5：`choice-token` 免训练 readout 的"打分—重归一—混合"数据流（`flowchart`）。
- 图 6：软目标 SFT 的歧义表达示意（`flowchart`）。

**总结**：这些实现如何共同支撑"低延迟 + 可校准 + 可控候选"的项目目标。

### 3.3 文件三：`/workspace/ReadCode/03-深刻观-哲学与升华.md`

**总述**：回望前两篇，提出"为什么是 System 1、为什么是委托、为什么是概率"三个追问。

**分述（三层次递进）**：
1. **设计哲学**：System 1 / System 2 双系统分工（Kahneman）在工程中的落地——把"选择"下沉为廉价的单次前向，把"推理"留给 LLM；"委托选择瓶颈，不是委托任务"（黄金法则、D0–D4 阶梯、负结果如 over-delegation）。
2. **框架思想**：候选空间即约束（bounded choice as a constraint）、概率即接口（probabilities as a first-class interface）、可校准优先于绝对正确、度量与协议前置（预声明任务集、冻结协议、配对不确定性区间）。
3. **核心主题升华**：从"模型即知识"到"模型即**决策的度量衡**"——JevAny 提供的是把不确定世界压缩成"可比较、可校准、可委派"的选择分布的通用接口。

**配图（≥3，实际给 4 张，均配图注）**：
- 图 1：双系统分工概念图（`flowchart`）。
- 图 2：委托阶梯 D0–D4 与任务层级 T0–T4 的两轴演进图（`flowchart LR` 或 `quadrantChart`，若用 `quadrantChart` 则严格最小化语法风险）。
- 图 3：核心洞见总结图（`mindmap` 或 `flowchart`，优先 `flowchart` 以降低语法风险）。
- 图 4：三篇递进关系收束图（`flowchart LR`）。

**总结（终极收尾）**：以**两三句话**概括整个 trilogy 的认知结晶（例如："JevAny 把决策从推理中剥离……"；需为最终定稿，届时精炼、朗朗上口、可独立传播）。

### 3.4 使用指南与案例（融入每篇末尾，不新增文件）
- **第 1 篇末尾 · 快速上手与端到端案例**：
  - 环境与安装（`pip install -e .` / `.[train]` / `.[serve,multimodal]`）。
  - 最小闭环四步：`jevany data init` → `data validate` → `train --config recipes/sft.toml` → `decide examples/request.json`；配套贴 [examples/request.json](file:///workspace/examples/request.json) 与真实字段。
  - 服务 + 客户端案例：`jevany serve …` + `JevClient.system_one(...)`（billing/shipping 工单路由，取自 [README.md](file:///workspace/README.md#L230-L253) 与 [API.md](file:///workspace/docs/API.md)）。
  - Playground 案例：`jevany demo` 的 FrozenLake / Doom / Crafter / Franka 臂（引用 `docs/demos/*.gif` 与 [PLAYGROUND.md](file:///workspace/docs/PLAYGROUND.md)）。
- **第 2 篇末尾 · 按技术点组织的可复现实验指南**：
  - 指针头残差消融（`--head-residual-dim 128`）、direct-token 基线（`--decision-mode lm_token`）、RLCR 续训（`--init-from … --rlcr`）、label-smoothing 自定义损失（[RESEARCH.md](file:///workspace/docs/RESEARCH.md)）。
  - 免训练 readout 评测命令（`jevany eval --readout choice`）。
  - 推理加速开关与测量（`--cuda-graphs` / `--fused-kernels`，以及 [scripts/benchmark_latency.py](file:///workspace/scripts/benchmark_latency.py)）。
  - 每个技术点绑定一个**真实案例**（含命令与引自 `results/*.json` 的数字），并注明"同名结论来自仓库文档"。
- **第 3 篇末尾 · 思辨型应用案例与决策清单**：
  - 何时该委托、何时不该（D0–D4 判定清单；over-delegation 反例：v1 `regex-log` 的 calls 20→30）。
  - 集成范式（[INTEGRATIONS.md](file:///workspace/docs/INTEGRATIONS.md)、`examples/tool_routing.py`、`examples/sql_repair.py`、`examples/service_recovery.py`）。
  - "企业落地检查表"：候选 2–4 个且有正确项、可逆性、低置信度回退、成对 reward/cost 论证。

### 3.5 论文检索与引用计划（第 2 篇，≥1 篇/技术点，真实且链接有效）
用 WebSearch/WebFetch 逐条核实标题、作者、年份与链接后再引用，格式：`[论文标题](URL) — 核心贡献一句话`。**待核实检索清单**（主题 → 预期论文）：
| 技术点 | 待检索论文主题 | 候选（需核实链接与年份） |
|---|---|---|
| RLCR 校准奖励 | correctness+proper scoring rule 的 RL | Damani et al., *Beyond Binary Rewards*, 2025 |
| 概率打分规则 | Brier / 严格proper scoring | Brier 1950；Gneiting & Raftery 2007 |
| 温度标定 | 现代 NN 校准 | Guo et al., *On Calibration of Modern Neural Networks*, 2017 |
| 软目标/蒸馏 | 软标签、分布匹配 | Hinton et al., *Distilling the Knowledge…*, 2015 |
| 指针/注意力打分 | Scaled Dot-Product Attention | Vaswani et al., *Attention Is All You Need*, 2017 |
| LoRA 适配 | 低秩适配 | Hu et al., *LoRA*, 2022 |
| 线性注意力主干 | Gated DeltaNet / DeltaNet / Mamba | Yang et al. 2024；Schlag et al. 2021；Gu & Dao 2023 |
| 融合/高效注意力 | FlashAttention | Dao et al., 2022 |
| 归一化/激活改进 | RMSNorm / SwiGLU | Zhang & Sennrich 2019；Shazeer 2020 |
| 块因果/序列打包 | packed sequence + block-causal mask | 相关训练效率工作（检索确定） |
| 受限解码/候选约束 | constrained / classification-head decoding | 相关 NLP 工作（检索确定） |

> 约束：**灰底为"检索后再定"，不预先假定 URL**。若某主题检索不到权威论文，则如实说明"以仓库文档为准"，不强行凑引用。

### 3.6 Mermaid 语法校验与纠错流程（对全部三篇的每一张图）
1. 写作时遵循"低风险语法子集"：优先 `flowchart`/`graph`、`sequenceDiagram`；节点标签统一**用双引号包裹**；标签内**禁用括号、`-->`、`#`、`;`、未转义 `<br>`**；`subgraph` 用 `subgraph id["标题"] … end` 且 id 不含空格；不使用 `%%` 注释与 HTML。
2. 每篇完成后执行**两步校验**：
   - 步骤 A（自动）：用脚本提取所有 ```` ```mermaid ```` 代码块，尝试用 `npx -y @mermaid-js/mermaid-cli`（`mmdc`）渲染为 SVG；渲染失败即定位行号。
   - 步骤 B（回退，若离线/无法安装）：运行自建 Python 校验脚本，检查：代码块数与标题类型合法性、`subgraph/end` 配对、括号/引号平衡、保留字冲突（`end`、`graph`、`class` 作节点名）、`classDef`/`style` 语法、箭头语法。
   - 两种途径都失败时，按"低风险语法子集"逐图人工复核并改写（这是用户明确要求的"对有问题者纠正"）。
3. 校验产物只作中间过程，**不写入 ReadCode/**；最终仅保留三篇 Markdown。
4. 记录一份"校验结果"写入各篇正文尾部的一句话脚注（如"本图已通过 mmdc 渲染校验"）——仅在成功时声明。

### 3.7 需要新读的仓库文件（执行阶段补充事实、确保引用真实）
- 第 1 篇：`docs/DEPLOYMENT.md`、`docs/TRAINING.md`、`docs/CASES.md`、`docs/INTEGRATIONS.md`、`jevany/cli.py`、`jevany/checkpoint.py`、`jevany/api.py`、`jevany/runtime.py`。
- 第 2 篇：`jevany/readout.py`、`jevany/inference.py`、`jevany/letter_*.py`、`jevany/data.py`、`jevany/datasets/build_sft.py`、`jevany/datasets/build_rlcr.py`、`docs/CHOICE_READOUT.md`、`docs/LETTER_READOUT.md`。
- 第 3 篇：`docs/experiments/AGENT_HARNESS_FRONTIER_PROTOCOL.md`（已读）、`docs/INTEGRATIONS.md`、`results/agent-harness-v1/*.md`、`reports/JevAny_Tech_Report_Agent_Harness_Appendix.md`。
> 以上读取均为只读，且**必须**在引用相应行号前完成，保证链接与数字准确。

---

## 四、假设与决策（Assumptions & Decisions）

| 项 | 决策 | 依据 |
|---|---|---|
| 输出目录 | `/workspace/ReadCode/`（项目根下） | 用户"保存在 ReadCode 文件夹" + 技能规范 |
| 文件数与命名 | 3 个：`01-整体观-项目全貌解读.md`、`02-具体观-算法与实现剖析.md`、`03-深刻观-哲学与升华.md` | 用户"用三篇讲述即可" + 技能命名规范 |
| 使用指南/案例 | **融入三篇末尾**，不新增第 4 文件 | 用户在 Phase 2 明确选择 |
| 语言/深度 | 中文正文 + 术语双语 + 论文级引用 | 用户在 Phase 2 明确选择 |
| 每篇图数 | ≥3 张 Mermaid（实际 5/6/4 张），图注必备 | 用户"每篇都要配图并说明" + 技能约束 |
| 事实来源 | 只引用仓库内真实文件/数字；论文须真实且链接有效 | 诚信原则 |
| 深挖模块 | 第 2 篇聚焦 6 大技术点（决策重构/指针头/direct-token/SFT软目标/RLCR/推理加速） | Phase 1 探索到的"最值得深挖"且文档支撑充分者 |
| 篇幅 | 允许长篇，深度优先（三篇合计预计 ≥ 1.5 万字量级） | 用户"可以长篇来写" |
| 不修改源码 | 仅新增 `ReadCode/` 下文档，不改动仓库任何源码/配置 | 任务性质为"阅读分析" |

---

## 五、验证步骤（Verification）

1. **完整性**：`LS /workspace/ReadCode` 确认 3 个 `.md` 存在且非空。
2. **结构合规**：逐篇核对——总述/分述/总结三段式；分述内层层递进；末尾含"使用指南与案例"节。
3. **图与图注**：统计每篇 Mermaid 代码块数 ≥3，且每块紧邻一句图注说明。
4. **Mermaid 语法**：执行 3.6 的两步校验（`mmdc` 渲染优先，回退脚本），逐图纠错直至全部通过；在正文记录校验结论。
5. **论文引用**：第 2 篇每个技术点 ≥1 条引用，抽查 3–5 条链接可达、标题/作者/年份一致。
6. **事实核对**：抽查关键数字（如加速比 3.72×、Transfer 86.04%、231/1046 项、LoRA rank 8、d_p=256、MAX_PACKED=2048）与仓库文档/代码一致；所有 `file:///` 链接指向真实存在的行号区间。
7. **收尾约束**：第 3 篇以**两三句话**作为终极概括收束，且可独立传播。
8. **交叉引用**：三篇之间形成显式递进引用（01→02→03 与回指），不重复堆叠。

---

## 六、执行顺序（建议）
1. 补读 3.7 列出的文件 → 固化事实与行号。
2. 检索并核实论文（3.5）→ 建立引用表。
3. 写 01 → 写 02 → 写 03（严格按 3.1/3.2/3.3 大纲）。
4. 每篇写完立即执行 Mermaid 两步校验并纠错（3.6），再做整篇验证（五）。
5. 三篇全部通过验证后，向用户交付（附 `computer://` 链接）。
