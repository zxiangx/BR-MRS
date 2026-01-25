# 方法
本节介绍BR-MRS,一个通过重构经典BPR Loss来显式建模模态不一致性的多模态推荐框架, 如图所示, BR-MRS首先遵循主流多模态推荐范式,利用图神经网络学习用户及物品表征. 在此基础上, BR-MRS引入两个核心组件, (i) 跨模态难负样本采样（CHNS），通过挖掘一个模态的混淆样本驱动另一模态提供判别性证据，从而利用信息性不一致；(ii) 协同感知 BPR 损失，通过约束融合表征的偏好边际显著优于任一单模态分支，从而抑制噪声性不一致导致的融合退化。 

## Graph-based Representation Learning

设用户集合 $\mathcal{U}$、物品集合 $\mathcal{I}$，观测交互 $\mathcal{O} \subseteq \mathcal{U} \times \mathcal{I}$。物品 $i$ 的模态特征记为 $\mathbf{x}_i^{(m)}$，$m \in \{t, v\}$。

**同质图构建与传播。** 为捕获同类实体间的潜在关联，我们构建物品同质图 $\mathcal{G}_{ii}$，其邻接权重综合交互共现与模态语义两类信号。具体地，物品对 $(i, j)$ 的边权定义为
$$
e_{ij} = \alpha \cdot \text{overlap}(\mathcal{N}_i^u, \mathcal{N}_j^u) + (1-\alpha) \sum\nolimits_{m} \beta_m \cos(\mathbf{x}_i^{(m)}, \mathbf{x}_j^{(m)}),
$$
其中 $\mathcal{N}_i^u$ 表示与物品 $i$ 交互的用户集合，$\alpha, \beta_m$ 为平衡系数。采用 top-$k$ 稀疏化以保留显著关联。用户同质图 $\mathcal{G}_{uu}$ 对称构建。经图卷积传播后，节点表征编码了同类实体间的行为模式与语义亲和性。

**异质图传播。** 在用户-物品二部图 $\mathcal{G}_{ui} = (\mathcal{U} \cup \mathcal{I}, \mathcal{O})$ 上，采用 LightGCN 执行 $L$ 层邻域聚合。各模态特征于独立信道中传播，最终表征由各层输出均值池化得到。具体地，对于物品 $i$，我们获得单模态表征 $\mathbf{q}_i^{(t)}, \mathbf{q}_i^{(v)} \in \mathbb{R}^d$ 及融合表征 $\mathbf{q}_i^{(f)} = \phi(\mathbf{q}_i^{(t)}, \mathbf{q}_i^{(v)})$；对于用户 $u$，我们对称地获得单模态偏好表征 $\mathbf{p}_u^{(t)}, \mathbf{p}_u^{(v)} \in \mathbb{R}^d$ 及融合表征 $\mathbf{p}_u^{(f)} = \phi(\mathbf{p}_u^{(t)}, \mathbf{p}_u^{(v)})$。用户-物品偏好得分定义为 $s_m(u,i) = \langle \mathbf{p}_u^{(m)}, \mathbf{q}_i^{(m)} \rangle$，$m \in \{t, v, f\}$。 



---
## 跨模态难负样本采样（CHNS）

在获得用户和物品的单模态表征及对应偏好得分后，我们进一步考虑如何显式地利用模态间的信息性不一致，以提升用户偏好的精准刻画。通过实证分析，我们观察到模态间的信息性不一致往往表现为跨模态混淆：对于同一正样本对 $(u,i^+)$，某些负样本在一个模态下得分较高，难以与正样本区分；而在另一模态下则得分显著更低，易于区分。此现象表明，一个模态的混淆样本能够有效地暴露另一模态的判别优势。

受经典 BPR 负采样策略的启发，我们提出跨模态难负样本采样（CHNS），即利用某一模态的混淆负样本，驱动另一模态显式贡献判别性证据。具体而言，对于每个正样本对 $(u,i^+)$，首先从未交互物品集合中抽取候选负样本池 $\mathcal{N}(u)$，然后分别在视觉和文本模态下选择得分最高的负样本 $i_v^- = \arg\max_{j\in\mathcal{N}(u)}s_v(u,j)$、$i_t^- = \arg\max_{j\in\mathcal{N}(u)}s_t(u,j)$，再将这些模态特异的负样本交叉交换用于另一模态分支的训练，从而得到跨模态的 BPR 损失：
$$
\mathcal{L}_{\mathrm{chns}}^v = -\sum_{(u,i^+)\in\mathcal{O}} \log\sigma\big(s_v(u,i^+)-s_v(u,i_t^-)\big),\quad
\mathcal{L}_{\mathrm{chns}}^t = -\sum_{(u,i^+)\in\mathcal{O}} \log\sigma\big(s_t(u,i^+)-s_t(u,i_v^-)\big).
$$

CHNS 通过上述跨模态监督机制，显式激活各模态的判别优势，将模态间的信息性不一致转化为有效的监督信号，以更精准地挖掘和利用模态互补信息。



## 协同感知 BPR 损失（Synergy-aware BPR Loss）

尽管CHNS能够有效利用模态间的信息性不一致，但噪声性不一致的存在可能导致多模态融合退化，即融合表征的成对排序能力劣于单模态分支。为此，我们基于BPR框架提出协同感知BPR损失，通过显式约束融合表征的偏好边际超越任意单模态分支，确保融合机制的鲁棒性。

具体而言，对于训练三元组 $(u,i^+,i^-)$（负样本 $i^-$ 随机采自未交互物品集合），定义融合及单模态的偏好边际为：
[
\Delta_f = s_f(u,i^+) - s_f(u,i^-),\quad
\Delta_t = s_t(u,i^+) - s_t(u,i^-),\quad
\Delta_v = s_v(u,i^+) - s_v(u,i^-).
]

已有研究表明，偏好边际的大小直接体现模型的排序置信度和区分能力，更大的偏好边际意味着更稳定的成对排序判别，并与推荐任务中的Top-$K$排序目标高度一致。因此，我们以表现最优的单模态为参照，引入严格的正边际约束 $\theta>0$，构建协同感知BPR损失：
[
\mathcal{L}*{\mathrm{syn}}
= -\sum*{(u,i^+,i^-)} \log\sigma\Big(\Delta_f - \max(\Delta_t,\Delta_v) - \theta\Big).
]

通过显式施加上述约束，即$\Delta_f > \max(\Delta_t,\Delta_v)+\theta$，协同感知损失有效抑制噪声模态的干扰，促使融合表征始终保持对任意单模态的优势，稳健提升多模态融合的可靠性与有效性。


## 总体目标

我们将跨模态难负样本采样损失和协同感知损失整合为 BR-MRS 的统一训练目标：

$$
\mathcal{L} = \lambda_h\,\mathcal{L}_{\mathrm{chns}} + \lambda_s\,\mathcal{L}_{\mathrm{syn}} + \lambda \lVert \Theta\rVert_2^2,
$$

其中 $\lambda_h$ 和 $\lambda_s$ 分别控制 CHNS 和协同感知损失的贡献，$\lambda$ 是正则化系数，$\Theta$ 表示所有可训练参数。



