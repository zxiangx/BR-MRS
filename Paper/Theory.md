%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Setup: scores and losses (BPR + InfoNCE + Orth)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Let $\phi_t,\phi_v$ be modality encoders and $\phi_f$ a fusion module. For each user $u$ we learn an embedding
$\mathbf{e}_u\in\mathbb{R}^d$. The modality-specific and fused scores are
\begin{equation}
s_t(u,i)=\langle \mathbf{e}_u,\mathbf{h}_t^i\rangle,\quad
s_v(u,i)=\langle \mathbf{e}_u,\mathbf{h}_v^i\rangle,\quad
s_f(u,i)=\langle \mathbf{e}_u,\mathbf{h}_f^i\rangle,
\end{equation}
where $\mathbf{h}_t^i=\phi_t(\mathbf{m}_t^i)$, $\mathbf{h}_v^i=\phi_v(\mathbf{m}_v^i)$, and
$\mathbf{h}_f^i=\phi_f(\mathbf{h}_t^i,\mathbf{h}_v^i)$.

We consider the commonly-used combined objective
\begin{equation}\label{eq:L_total}
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{BPR}}
+
\lambda_1\mathcal{L}_{\mathrm{InfoNCE}}
+
\lambda_2\mathcal{L}_{\mathrm{orth}},
\end{equation}
where the fused BPR loss is trained with negative sampling $i^-\sim q(\cdot\mid u)$:
\begin{equation}\label{eq:BPR}
\mathcal{L}_{\mathrm{BPR}}
=
-\mathbb{E}_{(u,i^+)\sim \mathcal{O}}
\ \mathbb{E}_{i^-\sim q(\cdot\mid u)}
\left[
\log\sigma\big(s_f(u,i^+)-s_f(u,i^-)\big)
\right].
\end{equation}
The (item-level) cross-modal alignment loss is
\begin{equation}\label{eq:InfoNCE}
\mathcal{L}_{\mathrm{InfoNCE}}
=
-\mathbb{E}_{i\sim \mathcal{I}}
\left[
\log
\frac{\exp\big(f(\tilde{\mathbf{h}}_t^i,\tilde{\mathbf{h}}_v^i)/\tau\big)}
{\sum_{j\in\mathcal{I}}\exp\big(f(\tilde{\mathbf{h}}_t^i,\tilde{\mathbf{h}}_v^j)/\tau\big)}
\right],
\end{equation}
where $\tilde{\mathbf{h}}_t^i,\tilde{\mathbf{h}}_v^i$ denote the embeddings used for contrastive learning
(e.g., after an $\ell_2$-normalization or a projection head), and $f(\cdot,\cdot)$ is a similarity.
The orthogonality regularizer is
\begin{equation}\label{eq:orth}
\mathcal{L}_{\mathrm{orth}}=\big\|\mathbf{H}_t^\top \mathbf{H}_v\big\|_F^2,\qquad
\mathbf{H}_t=[\mathbf{h}_t^1,\dots,\mathbf{h}_t^{|\mathcal{I}|}],\ \mathbf{H}_v=[\mathbf{h}_v^1,\dots,\mathbf{h}_v^{|\mathcal{I}|}].
\end{equation}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Definitions: unimodal ambiguity and fusion degradation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\begin{definition}[Modality-ambiguous negatives]\label{def:ambiguous}
Fix a user--positive pair $(u,i^+)$.
A negative item $j\notin\mathcal{O}_u$ is \emph{visual-ambiguous but text-disambiguatable} if
(i) the visual branch cannot reliably separate $i^+$ and $j$ (e.g., $\Delta_v(u,i^+,j)$ is small),
yet (ii) there exists task-relevant textual evidence that separates them.
We denote the set of such negatives by $\mathcal{A}_v(u,i^+)$.
\end{definition}

\begin{definition}[Fusion-degradation negatives]\label{def:degradation}
Fix $(u,i^+)$. A negative $j\notin\mathcal{O}_u$ is called a \emph{fusion-degradation negative} if
\begin{equation}
\Delta_t(u,i^+,j)>0,\quad \Delta_v(u,i^+,j)>0,\quad \text{but}\quad \Delta_f(u,i^+,j)\le 0,
\end{equation}
where $\Delta_m(u,i^+,j)\triangleq s_m(u,i^+)-s_m(u,j)$ for $m\in\{t,v,f\}$.
We denote this set by $\mathcal{D}(u,i^+)$.
\end{definition}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Assumption: rare exposure under negative sampling
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\begin{assumption}[Rare exposure under negative sampling]\label{ass:rare}
There exist constants $\rho_A,\rho_D\in(0,1)$ such that for a non-negligible fraction of $(u,i^+)\sim\mathcal{O}$,
\begin{equation}
q\big(\mathcal{A}_v(u,i^+)\mid u\big)\le \rho_A,
\qquad
q\big(\mathcal{D}(u,i^+)\mid u\big)\le \rho_D.
\end{equation}
That is, modality-ambiguous or fusion-degradation negatives occur with small probability under the
training negative sampler $q(\cdot\mid u)$.
\end{assumption}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Lemma: sampled BPR provides weak constraint on rare negatives
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\begin{lemma}[Sampled BPR weakly constrains rare negative subsets]\label{lem:bpr_rare}
Let $\ell(\Delta)\triangleq -\log\sigma(\Delta)$ and fix $(u,i^+)$. For any subset $S\subseteq \mathcal{I}\setminus\mathcal{O}_u$,
\begin{equation}
\mathbb{E}_{i^-\sim q(\cdot\mid u)}[\ell(\Delta_f(u,i^+,i^-))]
=
(1-p)\,\mathbb{E}_{i^-\sim q(\cdot\mid u),\,i^-\notin S}[\ell(\Delta_f)]
+
p\,\mathbb{E}_{i^-\sim q(\cdot\mid u),\,i^-\in S}[\ell(\Delta_f)],
\end{equation}
where $p=q(S\mid u)$. In particular, if $\ell(\Delta_f)\le M$ for all $i^-\in S$ (e.g., $\Delta_f\ge -B$ so $M=\log(1+e^B)$),
then the total contribution of constraints on $S$ is at most $pM$.
\end{lemma}
\begin{proof}
The decomposition is immediate by the law of total expectation conditioning on the event $\{i^-\in S\}$.
The bound follows since $\mathbb{E}[\ell(\Delta_f)\mid i^-\in S]\le M$ and thus $p\,\mathbb{E}[\ell(\Delta_f)\mid i^-\in S]\le pM$.
\end{proof}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Theorem 2: No guarantee for unimodal indistinguishability (unique evidence)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\begin{theorem}[No guarantee to resolve unimodal indistinguishability]\label{thm:no_unique}
Consider $\mathcal{L}_{\mathrm{total}}$ in \eqref{eq:L_total} trained with negative sampling \eqref{eq:BPR}.
Under Assumption~\ref{ass:rare}, there exist parameter settings $\Theta$ that attain arbitrarily small
$\mathcal{L}_{\mathrm{InfoNCE}}$ and $\mathcal{L}_{\mathrm{orth}}$, and achieve low sampled BPR loss,
yet fail to encode task-relevant modality-unique discriminative evidence needed to separate
$\mathcal{A}_v(u,i^+)$ (Definition~\ref{def:ambiguous}) for a non-negligible fraction of $(u,i^+)$.
Consequently, minimizing \eqref{eq:L_total} does not guarantee eliminating unimodal indistinguishability.
\end{theorem}

\begin{proof}
We give a constructive ``degenerate-solution'' argument that explicitly uses all three terms.

\paragraph{Step 1 (A representational family with shared and private channels).}
Assume the encoders can represent each item by
\[
\mathbf{h}_t^i=[\alpha\mathbf{c}_i;\ \mathbf{p}_t^i],\qquad
\mathbf{h}_v^i=[\alpha\mathbf{c}_i;\ \mathbf{p}_v^i],
\]
where $\mathbf{c}_i$ is a \emph{cross-modally alignable} (redundant/shared) factor and
$\mathbf{p}_t^i,\mathbf{p}_v^i$ are modality-private channels.
Let the fusion module ignore private channels, e.g.,
$\mathbf{h}_f^i = \mathbf{W}[\alpha\mathbf{c}_i;\alpha\mathbf{c}_i]$.
This family is standard in practice (a shared ``semantic'' backbone plus modality-specific residuals).

\paragraph{Step 2 (InfoNCE can be minimized without keeping unique evidence).}
Choose the contrastive embeddings to depend only on the shared factor, e.g.,
$\tilde{\mathbf{h}}_t^i=\mathrm{norm}(\mathbf{c}_i)$ and $\tilde{\mathbf{h}}_v^i=\mathrm{norm}(\mathbf{c}_i)$
(possibly via a projection head), so that matched pairs $(i,i)$ are maximally similar
while mismatched pairs are relatively dissimilar.
Then $\mathcal{L}_{\mathrm{InfoNCE}}$ can be driven arbitrarily close to its optimum by learning a discriminative
$\mathbf{c}_i$, regardless of what information is stored in $\mathbf{p}_t^i$.
In particular, any modality-unique evidence that is \emph{not} cross-modally alignable does not improve
\eqref{eq:InfoNCE} and thus is not required by this term.

\paragraph{Step 3 (Orthogonality can be satisfied by task-irrelevant private channels).}
Take $\{\mathbf{p}_t^i\}_i$ and $\{\mathbf{p}_v^i\}_i$ to be (approximately) uncorrelated across modalities,
e.g., independent isotropic noise with zero mean and sufficiently large dimension.
Then $\|\mathbf{H}_t^\top\mathbf{H}_v\|_F^2$ can be made arbitrarily small by (i) shrinking $\alpha$ so that the
shared block contributes negligibly to $\mathbf{H}_t^\top\mathbf{H}_v$, and (ii) choosing private channels whose
cross-covariance concentrates near zero.
Crucially, this reduces \eqref{eq:orth} without forcing the private channels to encode \emph{task-relevant} unique signals;
they may be purely nuisance variations.

\paragraph{Step 4 (Sampled BPR remains low without supervising ambiguity cases).}
Because $\mathbf{h}_f^i$ depends only on $\mathbf{c}_i$, the fused score $s_f(u,i)$ depends only on shared factors.
If most sampled negatives under $q(\cdot\mid u)$ are separable using shared factors, then BPR can be made small by
learning $\mathbf{e}_u$ and $\mathbf{c}_i$ to yield large margins on sampled triples.
Now focus on the ambiguity set $S=\mathcal{A}_v(u,i^+)$.
By Assumption~\ref{ass:rare}, $p=q(S\mid u)\le \rho_A$ for many $(u,i^+)$.
By Lemma~\ref{lem:bpr_rare}, the total BPR contribution from enforcing correct ranking over $S$ is at most $pM$,
which is negligible when $\rho_A$ is small (even if the model fails on $S$).

\paragraph{Step 5 (Failure on unimodal indistinguishability).}
By Definition~\ref{def:ambiguous}, negatives in $\mathcal{A}_v(u,i^+)$ are precisely those requiring
\emph{modality-unique} evidence to disambiguate.
Since the constructed fused scorer ignores the modality-unique channel, it need not separate $i^+$ from such $j$.
Hence unimodal indistinguishability can persist while all three loss terms remain low.

Combining Steps 2--4 yields a parameter family with low $\mathcal{L}_{\mathrm{InfoNCE}}$, low $\mathcal{L}_{\mathrm{orth}}$,
and low sampled $\mathcal{L}_{\mathrm{BPR}}$, yet without guaranteeing disambiguation on $\mathcal{A}_v(u,i^+)$.
Therefore the combined objective does not provide a guarantee. \qedhere
\end{proof}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Theorem 3: No guarantee for avoiding Fusion Degradation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\begin{theorem}[No guarantee to avoid fusion degradation]\label{thm:no_fd}
Consider \eqref{eq:L_total} in the sampled-BPR training regime \eqref{eq:BPR}.
Under Assumption~\ref{ass:rare}, the combined objective admits parameter settings $\Theta$ for which
$\mathcal{L}_{\mathrm{InfoNCE}}$ and $\mathcal{L}_{\mathrm{orth}}$ are small and the sampled BPR loss is low,
but there exists a non-negligible fraction of $(u,i^+)$ such that $\mathcal{D}(u,i^+)\neq\emptyset$
(Definition~\ref{def:degradation}). Hence minimizing \eqref{eq:L_total} does not guarantee preventing fusion degradation.
\end{theorem}

\begin{proof}
We again provide a constructive argument and explicitly invoke each term.

\paragraph{Step 1 (Unimodal branches rely on a robust shared signal).}
Let each item have a shared factor $\mathbf{c}_i$ that is predictive for ranking under most users.
Construct unimodal representations
\[
\mathbf{h}_t^i=[\mathbf{c}_i;\mathbf{0}],\qquad \mathbf{h}_v^i=[\mathbf{c}_i;\mathbf{0}],
\]
so both unimodal scores depend only on $\mathbf{c}_i$ and thus are robust:
for many negatives $j$, $\Delta_t(u,i^+,j)>0$ and $\Delta_v(u,i^+,j)>0$ whenever $\mathbf{c}_{i^+}$ is preferred to $\mathbf{c}_j$.

\paragraph{Step 2 (Inject a modality-private nuisance that enters fusion but is invisible to InfoNCE).}
Augment the visual branch with a private nuisance channel $\mathbf{n}_i$ that is \emph{task-irrelevant} but can be
propagated into fusion:
\[
\mathbf{h}_v^i=[\mathbf{c}_i;\ \mathbf{n}_i],\qquad
\mathbf{h}_t^i=[\mathbf{c}_i;\ \mathbf{0}],
\qquad
\mathbf{h}_f^i=\mathbf{W}_c\mathbf{c}_i+\mathbf{W}_n\mathbf{n}_i.
\]
Choose $\mathbf{W}_n\neq \mathbf{0}$ (a realistic situation in generic fusion modules such as concatenation + MLP).
Then there exist negatives $j$ with $\Delta_t>0$ and $\Delta_v>0$ (since unimodal branches ignore $\mathbf{n}$),
yet $\Delta_f(u,i^+,j)\le 0$ whenever the nuisance difference dominates the shared margin:
$\langle \mathbf{e}_u,\mathbf{W}_n(\mathbf{n}_{i^+}-\mathbf{n}_j)\rangle$ is sufficiently negative.
Such $j$ form the fusion-degradation set $\mathcal{D}(u,i^+)$.

\paragraph{Step 3 (InfoNCE remains low).}
Let the contrastive embeddings discard the nuisance channel, e.g.,
$\tilde{\mathbf{h}}_v^i=\mathrm{norm}(\mathbf{c}_i)$ and $\tilde{\mathbf{h}}_t^i=\mathrm{norm}(\mathbf{c}_i)$.
Then \eqref{eq:InfoNCE} depends only on $\mathbf{c}_i$ and can be optimized independently of $\mathbf{n}_i$.
Thus $\mathcal{L}_{\mathrm{InfoNCE}}$ can be small even though the fused scorer is sensitive to $\mathbf{n}_i$.

\paragraph{Step 4 (Orthogonality can be satisfied by choosing nuisance to be orthogonal across modalities).}
Take $\{\mathbf{n}_i\}_i$ to be approximately orthogonal to $\{\mathbf{c}_i\}_i$ in aggregate and uncorrelated with text features
(e.g., independent random vectors in high dimension). Then $\|\mathbf{H}_t^\top\mathbf{H}_v\|_F^2$ can be made small
because the cross-correlation between the text matrix (spanned by $\mathbf{c}_i$) and the visual nuisance subspace concentrates near zero.
Importantly, this reduces \eqref{eq:orth} while \emph{not} removing the nuisance influence on fusion.

\paragraph{Step 5 (Sampled BPR does not penalize rarely-sampled fusion-degradation negatives).}
Let $S=\mathcal{D}(u,i^+)$. By Assumption~\ref{ass:rare}, $p=q(S\mid u)\le \rho_D$ for many $(u,i^+)$.
Hence, by Lemma~\ref{lem:bpr_rare}, the expected BPR penalty associated with incorrectly ranking items in $S$ is at most $pM$,
which is negligible when $\rho_D$ is small.
As a result, training can drive the BPR loss low on the sampled negatives (typically separable by $\mathbf{c}_i$),
while leaving $\mathcal{D}(u,i^+)$ unresolved.

Combining the above steps, we obtain parameters with low $\mathcal{L}_{\mathrm{InfoNCE}}$, low $\mathcal{L}_{\mathrm{orth}}$,
and low sampled $\mathcal{L}_{\mathrm{BPR}}$, yet with $\mathcal{D}(u,i^+)\neq\emptyset$ for a non-negligible subset of users,
i.e., fusion degradation persists. Therefore, the combined objective provides no theoretical guarantee to prevent it. \qedhere
\end{proof}
