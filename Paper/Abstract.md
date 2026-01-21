



多模态推荐系统通过结合物品的视觉与文本等多模态信息, 已在个性化排序任务被广泛验证能够带来显著性能提升. 其中, 模态不一致性是多模态推荐系统面临的核心挑战, 例如, 视觉图像通常包含更细粒度的外观与风格信息, 而文本描述通常强调功能属性与使用场景.  直观上, 模态不一致性可以分解为两种成分: 信息性不一致, 即XXX, 噪声性不一致,即XXX.  

为了应对模态不一致, 早期方法使用LATE fUSION进行模态独立性建模, 但既XXX,又XXX. 而一些工作引入自监督学习,通过施加跨模态一致性损失, 在一定程度上缓解了噪声不一致的问题, 但是XXX.  近期, 一些工作开始尝试通过正交约束, 模态特异子空间等解耦机制, XXX, 但往往停留在几何解耦, 缺乏与推荐的个性化排序目标一致的显式约束, XXX. 

为了系统性分析现有工作在面对模态不一致的不足, 我们对现有的具有代表性的工作/CITE 展开实证分析, 图1和图2是实证分析的一部分, 实证结果揭示 (1) **现有方法对信息性不一致的利用不足**, 在一个模态无法准确刻画用户偏好的时候, 现有方法往往无法利用另一个模态的差异性信息(完善这个表述) ,  (2) 现有方法在噪声性不一致性的抑制上仍有缺陷, 特别的, 我们发现XXX.   

为了解决上述挑战, 我们提出XXX框架, 当一个模态不足以刻画用户偏好时, 利用另一个模态的不一致信息, 以建模信息性不一致. 同时, 在融合表征空间中施加





多模态推荐系统通过整合物品（item）的**视觉**与**文本**等异质模态信息，已在个性化排序任务中被广泛验证能够带来显著性能提升。不同模态往往包含不同方面的信息，例如，物品图像通常包含更细粒度的外观、材质与风格信息，而文本描述往往强调功能属性与使用场景，二者从不同视角刻画同一个物品，由此产生的信息差异与不一致性，被称为“modality inconsistency”。两个模态之间的差异可能在关键信息上相互补充，塑造更加具体的物品，也可能由于营销文案、信息缺失而引入使图文不一致的噪声。基于这一观察，我们将模态不一致性分为两类：一类是互补的、**能够增强正负物品区分能力**的 “informative inconsistency”，另一类是**会干扰用户偏好判别、破坏排序的”noisy inconsistency“**。这使得多模态推荐的核心挑战在于：在个性化排序目标下，模型既需要准确提取不同模态信息的 **informative inconsistency**，在单一模态不足以刻画用户偏好时实现有效互补；又需要避免将 **noisy inconsistency** 引入融合表征，从而防止其干扰物品表征并削弱推荐性能。

围绕“如何处理模态间不一致性”这一问题，已有研究大体沿着几条路线推进：早期方法往往采取模态独立建模并依赖 late fusion，在不同模态上分别学习打分/排序分支，再在决策阶段融合；这在一定程度上降低了噪声的直接传播，但也弱化了跨模态协作带来的增益，使模型在必须依赖”informative inconsistency"才能准确区分的困难负样本上受限。进一步地，许多工作引入自监督学习（例如对比式对齐或生成式自监督），通过施加跨模态一致性约束，这类方法在一定程度上缓解了 noisy inconsistency 带来的不稳定性，但其训练目标通常更强调一致性本身，因而难以显式鼓励 informative inconsistency 在排序中的判别作用。近期也有方法意识到informative inconsistency的价值，尝试通过正交约束、模态特异子空间等显式解耦机制保留这类inconsistency；然而，这类设计多数停留在表征空间的几何解耦层面，缺乏与个性化排序目标相一致的优化约束，难以保证 informative inconsistency 能够被有效保留。

为了系统地刻画上述不足，我们进一步开展实证分析，并发现现有方法在处理跨模态不一致信息时常出现两类普遍的失效现象。第一，模型常将与正样本 item 在**某一模态**（如文本或图像）上高度相似的负样本【排得靠前】，尽管该负样本在**另一模态**上其实存在清晰的区分线索——这表明模型未能在排序中有效“调用”另一模态的互补证据来纠正混淆。第二，在部分用户上，融合后的多模态表征反而弱于某个单模态表征，出现【融合退化现象】（参考文章），表明融合表征可能被噪声稀释，从而削弱了对用户偏好的判别力。【这里可以加一下PID理论，我们显式将信息分成什么什么部分】

为解决上述两类挑战，我们提出 **BR-MRS** 这一多模态推荐框架。首先，我们引入 **Cross-modal Hard Negative Sampling（CHNS）**，通过跨模态地构造困难负样本，使模型在训练过程中被显式驱动去利用不同模态之间对排序有判别力的差异，从而更充分地挖掘 **informative inconsistency**。其次，我们提出 **Synergy-aware BPR Loss** 来缓解融合退化问题，在融合分支训练中以单模态分支作为参照，持续约束融合表征在正负 item 区分上优于任一单模态，从目标层面抑制 **noisy inconsistency** 对融合表征的干扰。【这个地方放一些实验结果说明方法的有效性】

Our contributions are summarized as follows:

1. **Modality Inconsistency Diagnosis.**
    We identify and analyze the limitations of existing multimodal recommender systems in exploiting informative inconsistency and suppressing noisy inconsistency, and show how these limitations lead to cross-modal complementarity failure and fusion degeneration.

   **A Synergy-Oriented Training Framework.**
    We propose **BR-MRS**, which combines cross-modal hard negative sampling to activate informative inconsistency with a synergy-aware ranking objective that constrains the fused representation to outperform unimodal branches.

   **Strong Empirical Performance.**
    Extensive experiments on multiple benchmarks demonstrate that **BR-MRS** consistently outperforms state-of-the-art methods, with ablations and case studies validating the effectiveness of each component.