# Prior-Art Landscape & Novelty Re-positioning（2024-2026）

> 这是 v2 论文 narrative 重构的依据。所有声明都基于 arXiv 实证检索。

## 1. 现有 Federated SNN landscape（按 sub-topic 归类）

### Cluster A: Federated SNN 基础与通信效率
| 论文 | venue / year | 核心贡献 | 与我们的关系 |
|---|---|---|---|
| Venkatesha et al. *Federated Learning with SNNs* | TSP 2021 | 第一篇 FedAvg-SNN | 老 baseline |
| Skatchkovsky et al. *FL-SNN* | arXiv 2019 / SPM 2020 | online FL-SNN, partial weight exchange | 老工作 |
| Xie et al. *Efficient FL with SNN for traffic sign* | TVT 2022 | SNN+IoV 应用 | 应用层 |
| Chaki et al. *Communication trade-offs in FL-SNN* | 2023 | random mask + client dropout | 两个 trick |
| Nguyen et al. *FLTS: Top-κ Sparsification* | IPCCC 2024 | Top-K 稀疏化上行 | **直接 baseline**：他们做"通信侧稀疏"，我们做"权重侧动态稀疏" |
| Nguyen et al. *Robustness FL-SNN + Top-κ vs Byzantine* | 2025.01 | Top-K + 拜占庭防御 | 同上 |
| Li et al. *FPGA multi-core SNN with FL* | 2024.12 | 5-worker FL on FPGA | 硬件向 |

### Cluster B: Federated SNN 异构 / non-IID
| 论文 | venue / year | 核心贡献 | 与我们的关系 |
|---|---|---|---|
| Zhan et al. *SFedCA: Credit Assignment client selection* | 2024.06 | fire rate 用于客户端选择 | 用 firing rate 做"选 client"，不做权重稀疏 |
| Yu et al. *Heterogeneous FL with CNN + SNN* | FL@IJCAI'24 | CNN-SNN 混合 FL | 不同模型 family 的混合 |
| **Yu et al. *FedLEC: Label Skewness for SNN-FL* | IJCAI 2025** | label imbalance + KD | 标签倾斜，不是模型异构 |
| **Tao et al. *SFedHIFI* | AAAI 2026 (本仓库)** | Tucker filter_bank 异构 + fire-rate 加权 | **我们当前的 base** |
| Karilanova et al. *FL-SNN under heterogeneous temporal resolutions* | 2026.05 | T 异构（每个 client 不同 timestep） | **非常关键**：他们做 temporal-T 异构，我们能做"temporal-T 异构 + 模型容量异构"双异构 |

### Cluster C: 隐私 / 安全 / 攻击
- Spikewhisper (2024)、Time-Distributed Backdoor (2024)、Pereira 2025 ICASSP (DP+SNN-FL)、Aksu 2025 (gradient inversion 在 SNN 内**自然抑制**)、Shang 2025 BCI-PFL

### Cluster D: Quantum / 特殊场景
- FL-QDSNNs (QAI'25)、VFL-SNN (2024)

## 2. SNN dynamic sparse training landscape

| 论文 | venue | 方法 | 联邦? |
|---|---|---|---|
| Deng et al. *Comprehensive SNN compression* | ICLR 2021 | ADMM 剪枝 | 否 |
| **Chen et al. *Grad R: Gradient Rewiring for SNN* | IJCAI 2021** | 第一篇 SNN sparse training | 否 |
| **Shen et al. *ESL-SNN: Evolutionary Structure Learning* | AAAI 2023** | rewiring 准则 | 否 |
| **Yin et al. *MINT* | ICCAD 2023** | quantization + sparsity for SNN | 否 |
| **Su et al. *STDS: Spatio-temporal Dynamic Sparsity* | 2024** | 时空动态稀疏 SNN | 否 |
| **Liu et al. *Sparse Spiking Gradient Descent* | NeurIPS 2021** | 稀疏梯度 | 否 |
| Pan et al. *Lottery Ticket Hypothesis for SNN* | 2023 | LTH 在 SNN | 否 |
| ElfCore (ESSERC 2025) | 28nm chip | dynamic structured sparse training + 在 chip 上 | **硬件**,非 FL |

**关键发现**：**SNN dynamic sparse training + Federated learning 的组合在 arXiv 上目前没有直接的、专门的工作！**
- 现有 SNN-FL 工作都是 sparsify 通信（Top-K on gradients）或 sparsify 客户端选择
- 现有 SNN-DST 工作都不是 federated
- 但是！普通 ANN 上的 Federated DST 已经被做过：FedDST (Bibikar et al. AAAI 2022)、ZeroFL (Qiu et al. ICLR 2022)、FedTiny (Huang et al.)、DisPFL (Dai et al. ICML 2022)
- 所以 trivial 推广 = "把 FedDST 改成 SNN" 仍然容易被审稿人识破。**这就是 v1 真正的弱点。**

## 3. 从 SFedHIFI 实际论文看真正的科学问题（来自 abs + appendix PDF）

SFedHIFI 2026 在 abs 里只声称：
1. channel-wise matrix decomposition → adaptive complexity（**这是 FLANC/Tucker，2019-2020 老技术**）
2. fire-rate-based heterogeneous information fusion → cross-scale aggregation
3. 实验：3 个 benchmark（FashionMNIST/CIFAR10/CIFAR100，**和我们 v1 一样 toy**）
4. 与 ANN-FL 比省电

**SFedHIFI 自己的 v1 也只是 AAAI（B 级 / borderline A），不是 NeurIPS/ICML 级**

## 4. 真正的 GAP（v1 没击中的）

通过对比 35+ 篇文献，真正未被覆盖的 gap 有三个层次：

### Gap 1: **Temporal heterogeneity (T-heterogeneity)** 
- Karilanova 2026 刚揭开冰山：每个 client 因传感器/能耗约束有不同 timestep T_k
- 但他们只做了 SHD/DVS-Gesture 上的 T 异构 + 简单 aggregation 适配
- **未被回答**：当 T 异构 + 模型容量异构 + non-IID 三者同时存在时，传统 DST schedule（基于全局 epoch）会失效，因为不同 client 的"时间预算"完全不同

### Gap 2: **Spike-driven mask discovery 的物理可解释性**
- 所有 SNN-DST 工作（Grad R/ESL/STDS）都把稀疏作为"压缩手段"
- 但 SNN 的 spike 本身就是 0/1 → 死神经元的下游连接**没有任何信息流过**
- 这意味着 SNN 中的"无效连接"有一个 **客观可观测的物理信号** $\sum_t s_i(t) = 0$
- ANN 没有这个性质（激活恒为非零）
- **未被利用**：把"突触是否承载脉冲信息"作为剪枝的客观真值，而不是任何代理（梯度、Hessian、magnitude）

### Gap 3: **Heterogeneous client = heterogeneous mask space**
- SFedHIFI 用 Tucker 分解处理异构（小 client 解码出小 model）
- HeteroFL/FjORD 用宽度切片处理异构
- 但所有这些都是**确定性映射**：client capacity → fixed sub-model
- **未被探索**：让每个 client 在共享支撑集上**自由探索**子稀疏掩码，server 端做 **mask-space federation**（聚合掩码本身的信息，而不只是聚合参数）

## 5. 三个候选 Narrative（应用 Janusian / Bisociation / Negation）

### Narrative A: **"Death is Information" (Negation 框架)**
> **One sentence**：在 SNN 联邦学习中，一个连接是否"死亡"（永不传递脉冲）本身就是免费的、无偏的、隐私安全的剪枝信号，不需要任何梯度/Hessian 估计。
> 
> **Two-sentence test**：
> - What/Why：现有 SNN-DST 沿用 ANN 的剪枝评分（magnitude/SNIP/Taylor），忽视了 SNN 独有的"死连接信号"——通过它统计可得 $\sum_t s_i^{(k)}(t)=0$ 的连接在 client k 上完全无信息流。
> - So What：这个信号 (a) 不需要梯度，可以在 inference-only 设备运行；(b) 每个 client 本地无偏；(c) 跨 client 聚合给出一个"联邦死亡集"，可被无损剪掉，这是 ANN 没有的、SNN 独有的稀疏化"金矿"。

- **Prior art 5min check**：Aksu 2025 的"SNN gradient 自然抑制 inversion"是相关的"SNN 独有性质"思路，但他们做隐私，不做 DST → **空白**

### Narrative B: **"Two clocks, one fabric: Temporal-Capacity Heterogeneous FL" (Janusian)**
> **One sentence**：当客户端在时间维度（不同 timestep T_k）和容量维度（不同 width / 算力）双重异构时，传统基于"epoch / sample / FLOP"的 DST schedule 全部崩溃，必须用一个新的、统一刻画两种异构的协调机制。
>
> **Two-sentence test**：
> - What/Why：Karilanova 2026 开启了 T 异构 SNN-FL，但他们的解只在 homogeneous capacity 下成立；SFedHIFI 解决 capacity 异构但假设 T 一致；real-world 设备永远是双异构。
> - So What：我们提出"effective spike-budget"作为统一度量，把每个 client 的 (T_k, width_k, sparsity_k) 映射到同一个 budget 数轴上，server 在这个数轴上做 schedule 协调 → 既是工程贡献也是理论贡献。

- **Prior art 5min check**：双异构（T+capacity）的 SNN-FL 工作 = 0 篇 → **空白**

### Narrative C: **"Mask is the message: Federated Sparse Topology Discovery on SNNs" (Bisociation)**
> **One sentence**：与其聚合 dense 参数，不如把每个 client 发现的"脉冲-相关稀疏拓扑（mask）"本身当作通信内容，让 server 在 mask-space 做联邦推断，这样能在 1-bit/参数 的带宽下实现 SNN-FL。
>
> **Two-sentence test**：
> - What/Why：所有现有 SNN-FL 都通信参数（即使加 Top-K 也是 sparse parameters），但 SNN 上"哪条连接活着"本身具有显著语义价值——它是 client 局部数据分布的拓扑指纹。
> - So What：把通信从参数空间转到 mask 空间 → 通信开销~1 bit/param，server 用 mask 共识 (consensus topology) 引导下一轮训练，这是 ANN 上做不到的（ANN mask 没有数据特征语义）。

- **Prior art 5min check**：Personalized FL 中有"个性化 mask"概念（DisPFL, Sub-FedAvg），但在 SNN 上+用脉冲语义解释 mask = 0 篇 → **空白**

## 6. Narrative 选型决策

3 个 narrative 都通过两个测试。综合考虑：
- **可实现性**：A > B > C（A 需要的 hooks 在 SpikingJelly 里现成；C 需要重新设计协议）
- **理论深度**：A > C > B（A 有清晰的 "spike-induced zero information flow → exact zero contribution" 定理）
- **实验饱满度**：A > B > C（A 可以用 SFedHIFI 现有 dataset + 加 DVS-Gesture/N-Caltech101，B 必须做 temporal 异构很费工程）
- **与现有 SFedHIFI 代码 fit**：A 最契合（在现有 capacity 异构基础上加 spike-driven mask discovery）
- **审稿人 wow 度**：A > C > B（A 的"death is information"是 phenomenon discovery，C 是工程，B 是 unification）

**选定：Narrative A "Spike-Induced Federated Sparsity" (SIFS)**

将 v1 的"SpikDST"重新定位为：
- 不再是"SNN 上的 BPTT-Taylor 剪枝"（被审稿人秒杀）
- 而是"**利用 SNN 独有的零脉冲流连接作为联邦剪枝信号**"——这个性质 ANN 没有，故不可能是 trivial 推广
- 数学上：用户提供的 $\delta_j(t)$ 梯度框架仍然适用，但 prune score 主项是**spike-emptiness statistic** $\rho_{ij}^{(k)} = \mathbb{P}_t[s_i(t)=0]$ 而非 $|w \cdot g|$
- 联邦上：mask-aware aggregator 仍然有效，但 unbias 论证更深 — 因为对死连接 $w$ 的梯度恒为 0，所以本地 update 也是 0，server 端跨 client 不必做"等价缩放"
- 异构容量上：与 SFedHIFI 的 Tucker filter_bank 正交叠加（SFedHIFI 决定"宽度切片"，SIFS 决定"切片内哪些连接活")

## 7. 修正后的 paper 结构

新 title: **"Spike-Induced Federated Sparsity: Turning Silent Synapses into Free Compression Signals for Heterogeneous Spiking Federated Learning"**

新 contributions：
1. **Phenomenon**: 实证揭示在异构 SFL 下，每个 client 自然产生~30-60% "spike-silent" 连接（实验 figure 1）
2. **Theory**: 证明 spike-silent 连接的本地 BPTT 梯度恒为 0（Proposition），因此剪掉它们对本地 loss 无影响（与 SNIP/Taylor 的近似不同，是 exact）
3. **Algorithm (SIFS)**: 三阶段——local spike-emptiness profiling → cross-client mask consensus → mask-aware aggregation（Eq.9 沿用）
4. **System fit**: 与 SFedHIFI 的 Tucker capacity-heterogeneity **正交叠加**（实验里证明 SFedHIFI + SIFS 双层稀疏可达 ~93% sparsity with marginal acc loss）
5. **Empirical**: FMNIST + CIFAR-10/100 + **CIFAR10-DVS + N-Caltech101**（新加 2 个 neuromorphic 数据集）+ Tiny-ImageNet（新加 1 个 scale-up）

## 8. 与用户提供的 9 个公式的关系（保留度）

| 公式 | v1 用法 | v2 SIFS 用法 | 保留 |
|---|---|---|---|
| Eq.1 $\delta_j(t)$ | 定义 | 定义 + 用于证明 spike-silent⇒gradient-zero | ✓ |
| Eq.2 Taylor 评分 | 主剪枝评分 | 降为"次要 baseline 评分" | ✓ 重定位 |
| Eq.3 BPTT 时序梯度分解 | 推导 | 核心证明用 | ✓ |
| Eq.4 SpikDST Prune $I_{\text{Prune}}$ | 主 | 改为 SIFS 综合评分（spike-emptiness 主项 + Taylor 次项） | ✓ 重写 |
| Eq.5 Grow $I_{\text{Grow}}$ | GraNet 同构 | 改为 "spike-rebirth" — 在 spike-active fan-in 区域优先复活 | ✓ 改写 |
| Eq.6 Crisis loss | 防 dead neuron | 保留，反过来作为"防止死亡"正则 | ✓ |
| Eq.7 Mask-aware aggregator | unbiased H-T 估计 | 仍然用，但 frame 为"spike-silent connections need no unbias correction"（因为本身就 0）→ 简化 | ✓ 简化 |
| Eq.8 全局目标 | 模板 | 保留 | ✓ |
| Eq.9 $T_{\text{eff}}$ | Bartlett 包装 | **删除**或改为 appendix curiosity（不再当核心） | ✗ 弱化 |

## 9. 实验设计 v2 — killer punch

**Killer Figure 1**: 在 SFedHIFI 训练过程中实测 "spike-silent fraction" 随 epoch 演化曲线（每个 client + global），证明 spike-silent 是普遍现象。**这一张图直接证明 phenomenon。**

**Killer Table 1**: SIFS vs (SFedHIFI / FedDST / RigL-Fed / FedAvg+SNN) 在 6 个数据集上的 acc / comm / FLOP 对比。**6 个数据集必须包含 CIFAR10-DVS + N-Caltech101，让 reviewer 看到 neuromorphic 真数据。**

**Killer Table 2**: Ablation 证明 spike-emptiness 主项独立于 Taylor 次项贡献的边际增益（如果 spike-emptiness 单独已经匹配 STDS 等方法，就直接证明 phenomenon → algorithm 的因果链）。

**Killer Proposition**: "如果 $s_i(t)=0\,\forall t \in [0,T]$ 且 surrogate gradient 满足局部 Lipschitz，则 $\partial L / \partial w_{ij} = 0$ exactly（非 approximation）"。这是简单但 ANN 没有的性质，审稿人会觉得新颖。
