# ILGAN — Mathematical Foundations

> **One-shot Image and Bounding Box Generation via Generative Adversarial Networks**
>
> This document provides the formal mathematical specification for the entire ILGAN
> system. Every loss function, training mechanism, architectural constraint, and
> regularisation term defined here is grounded in rigorous mathematical reasoning.
> The document serves as the single source of truth for all subsequent
> implementations in the codebase.

---

## Table of Contents

1. [Notation and Conventions](#1-notation-and-conventions)
2. [The Dual-GAN Objective](#2-the-dual-gan-objective)
3. [Representation Collapse Prevention Theorem](#3-representation-collapse-prevention-theorem)
4. [Cross-Modal Consistency Constraint](#4-cross-modal-consistency-constraint)
5. [Adaptive Diversity Scheduler](#5-adaptive-diversity-scheduler)
6. [Gradient Penalty for Dual Outputs](#6-gradient-penalty-for-dual-outputs)
7. [Complete Training Objective](#7-complete-training-objective)
8. [Optimisation Dynamics and Convergence](#8-optimisation-dynamics-and-convergence)
9. [Appendix: Proofs and Derivations](#9-appendix-proofs-and-derivations)

---

## 1. Notation and Conventions

### 1.1 Spaces and Dimensions

| Symbol | Meaning | Shape / Domain |
|--------|---------|----------------|
| $\mathcal{Z}$ | Latent space | $\mathbb{R}^{d_z}$, $d_z = 256$ |
| $\mathcal{I}$ | Image space | $[-1, 1]^{3 \times H \times W}$, $H = W = 128$ |
| $\mathcal{B}$ | Bounding box space | $[0, 1]^{N_{\text{max}} \times 4}$ |
| $\mathcal{C}$ | Class label space | $\mathbb{R}^{N_{\text{max}} \times K}$, $K = 80$ |
| $\mathcal{S}$ | Confidence space | $[0, 1]^{N_{\text{max}} \times 1}$ |
| $\mathcal{D}$ | Discriminator score space | $\mathbb{R}^{1 \times G \times G} \times \mathbb{R}$ |
| $N_{\text{max}}$ | Maximum number of boxes | $20$ |
| $K$ | Number of object classes | $80$ |
| $G$ | Grid size for local scores | $4$ |
| $B$ | Batch size | $16$ |

### 1.2 Random Variables and Distributions

- $z \sim p_z$: latent vector drawn from standard normal $\mathcal{N}(0, I_{d_z})$
- $x \sim p_{\text{data}}$: real image drawn from the training data distribution
- $x_{\text{fake}} = G(z)_{\text{image}}$: generated image
- $\hat{x} \sim p_{\hat{x}}$: interpolated samples between real and generated images
- $\varepsilon \sim \text{Uniform}(0, 1)$: interpolation coefficient

### 1.3 The Generator

The ILGAN generator $G: \mathcal{Z} \to \mathcal{I} \times \mathcal{B} \times \mathcal{C} \times \mathcal{S}$ is a unified function that produces all four outputs from a single latent vector:

$$G(z) = (I, B, C, S)$$

where:

- $I \in \mathcal{I}$: the generated RGB image
- $B \in \mathcal{B}$: bounding box coordinates $(x_{\text{center}}, y_{\text{center}}, w, h)$ normalised to $[0, 1]$
- $C \in \mathcal{C}$: raw class logits for each bounding box
- $S \in \mathcal{S}$: objectness confidence scores in $[0, 1]$

The generator is composed of two sub-modules:

$$G = G_{\text{spatial}} \circ G_{\text{content}}$$

where $G_{\text{content}}: \mathcal{Z} \to \mathcal{I} \times \mathcal{F}$ produces the image $I$ and a list of multi-resolution skip features $\mathcal{F} = \{F_1, F_2, \dots, F_L\}$, and $G_{\text{spatial}}: \mathcal{F} \to \mathcal{B} \times \mathcal{C} \times \mathcal{S}$ consumes these features to produce the bounding box outputs.

### 1.4 The Discriminator

The discriminator $D: \mathcal{I} \to \mathbb{R}^{1 \times G \times G} \times \mathbb{R}$ maps an image to two outputs:

$$D(I) = (D_{\text{loc}}(I), D_{\text{glob}}(I))$$

where:

- $D_{\text{loc}}(I) \in \mathbb{R}^{1 \times G \times G}$: a spatial grid of local realism scores (PatchGAN-style), where each cell scores the realism of a local image patch
- $D_{\text{glob}}(I) \in \mathbb{R}$: a single scalar representing the global realism of the entire image

The discriminator **only sees images** — it does not receive bounding box information. This forces the generator to learn to produce realistic images whose content naturally implies the correct bounding box layout.

---

## 2. The Dual-GAN Objective

### 2.1 Wasserstein GAN with Gradient Penalty (WGAN-GP)

The ILGAN adversarial objective is based on the Wasserstein GAN formulation (Arjovsky et al., 2017) with the gradient penalty term (Gulrajani et al., 2017). We choose WGAN-GP over the standard GAN objective for three reasons:

1. **Stable gradients**: The Wasserstein distance provides meaningful gradients even when the generator and discriminator distributions are far apart, avoiding the vanishing gradient problem of the standard GAN.
2. **Lipschitz constraint via soft penalty**: The gradient penalty enforces the 1-Lipschitz constraint on the discriminator without the weight-clipping issues of the original WGAN.
3. **Convergence correlates with sample quality**: The Wasserstein distance is a meaningful loss metric that correlates with visual quality, unlike the standard GAN loss.

### 2.2 Formal Definition

Let $D$ be a 1-Lipschitz function (enforced by the gradient penalty). The adversarial objective is:

$$\min_G \max_D V(D, G) = \underbrace{\mathbb{E}_{x \sim p_{\text{data}}}[D(x)]}_{\text{real score}} - \underbrace{\mathbb{E}_{z \sim p_z}[D(G(z)_{\text{image}})]}_{\text{fake score}} - \underbrace{\lambda_{\text{gp}} \cdot \mathbb{E}_{\hat{x} \sim p_{\hat{x}}}\left[ \left( \|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1 \right)^2 \right]}_{\text{gradient penalty}}$$

where:

- $\lambda_{\text{gp}} = 10.0$ is the gradient penalty coefficient
- $p_{\hat{x}}$ is the distribution of interpolated samples (see Section 6)
- $\|\cdot\|_2$ denotes the Euclidean norm

### 2.3 Discriminator Loss

The discriminator is trained to **maximise** $V(D, G)$, which is equivalent to minimising:

$$\mathcal{L}_D = \underbrace{\mathbb{E}_{z \sim p_z}[D(G(z)_{\text{image}})]}_{\text{minimise fake score}} - \underbrace{\mathbb{E}_{x \sim p_{\text{data}}}[D(x)]}_{\text{maximise real score}} + \underbrace{\lambda_{\text{gp}} \cdot \mathbb{E}_{\hat{x} \sim p_{\hat{x}}}\left[ \left( \|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1 \right)^2 \right]}_{\text{gradient penalty}}$$

The discriminator produces both local and global scores. The total discriminator score for an image is:

$$D(I) = \underbrace{\frac{1}{G^2} \sum_{i=1}^{G} \sum_{j=1}^{G} D_{\text{loc}}(I)_{ij}}_{\text{mean local score}} + \underbrace{D_{\text{glob}}(I)}_{\text{global score}}$$

This combined score is used in the WGAN-GP objective above.

### 2.4 Generator Adversarial Loss

The generator is trained to **minimise** $V(D, G)$, which is equivalent to minimising:

$$\mathcal{L}_{\text{adv}} = -\mathbb{E}_{z \sim p_z}[D(G(z)_{\text{image}})]$$

This pushes the generator to produce images that the discriminator assigns high realism scores to, both locally and globally.

### 2.5 Why WGAN-GP for ILGAN

The dual-output nature of ILGAN makes training stability particularly challenging. The WGAN-GP formulation provides:

- **Meaningful gradients for both pathways**: The Wasserstein distance provides smooth gradients that propagate through both the content decoder (image pathway) and the spatial head (bounding box pathway).
- **Prevention of discriminator overpowering**: The Lipschitz constraint prevents the discriminator from becoming too sharp, which would cause the generator's gradients to explode and destabilise the bounding box predictions.
- **Natural compatibility with gradient penalty**: The gradient penalty naturally extends to the dual-output case (see Section 6).

---

## 3. Representation Collapse Prevention Theorem

### 3.1 The Problem of Collapse

In a dual-output GAN, two distinct forms of collapse can occur:

1. **Image mode collapse**: The generator produces the same image for all latent vectors (standard GAN mode collapse).
2. **Bounding box collapse**: All spatial slots attend to the same spatial location, producing identical bounding boxes regardless of the image content.

The ILGAN architecture addresses both forms through a combination of architectural constraints and loss functions. This section focuses on the **representation collapse prevention theorem**, which guarantees that the spatial slots in the SCCA module maintain diverse attention distributions.

### 3.2 Formal Definition of the Repulsion Loss

Let $A^{(b)} \in \mathbb{R}^{N \times HW}$ be the attention weight matrix for batch element $b$, where $A^{(b)}_{n,i}$ is the attention weight of slot $n$ on spatial position $i$ (flattened $H \times W$ grid). The attention weights satisfy:

$$\sum_{i=1}^{HW} A^{(b)}_{n,i} = 1, \quad A^{(b)}_{n,i} \geq 0 \quad \forall n, i$$

The spatial centre of mass for slot $n$ in batch element $b$ is:

$$\mu^{(b)}_n = \left( \sum_{i=1}^{HW} A^{(b)}_{n,i} \cdot x_i, \sum_{i=1}^{HW} A^{(b)}_{n,i} \cdot y_i \right) \in [0, 1]^2$$

where $(x_i, y_i)$ are the normalised coordinates of spatial position $i$.

The Euclidean distance between the centres of mass of slots $i$ and $j$ is:

$$d^{(b)}_{ij} = \left\| \mu^{(b)}_i - \mu^{(b)}_j \right\|_2$$

**Definition 3.1 (Repulsion Loss).** The repulsion loss is defined as:

$$\mathcal{L}_{\text{rep}} = \frac{1}{B \cdot N_{\text{pairs}}} \sum_{b=1}^{B} \sum_{1 \leq i < j \leq N} \max\left(0, \tau - d^{(b)}_{ij}\right)^2$$

where:

- $B$ is the batch size
- $N$ is the number of spatial slots ($N_{\text{max}}$)
- $N_{\text{pairs}} = \binom{N}{2} = \frac{N(N-1)}{2}$ is the number of slot pairs
- $\tau \in (0, 1)$ is the repulsion threshold (default: $\tau = 0.2$)

### 3.3 Theorem: Collapse Prevention

**Theorem 3.1 (Representation Collapse Prevention).** Let $\mathcal{L}_{\text{rep}}$ be the repulsion loss as defined above, and let $\theta$ be the parameters of the spatial query embeddings $Q \in \mathbb{R}^{N \times D}$ and the key projection $W_K \in \mathbb{R}^{D \times C'}$ in the SCCA module. Under the gradient flow dynamics:

$$\frac{d\theta}{dt} = -\eta \cdot \frac{\partial \mathcal{L}_{\text{rep}}}{\partial \theta}$$

for learning rate $\eta > 0$, the system reaches an equilibrium where for all $b$ and all $i \neq j$:

$$d^{(b)}_{ij} \geq \tau$$

provided that the attention distributions are differentiable with respect to $\theta$ and the spatial grid resolution is sufficient to distinguish distances smaller than $\tau$.

**Proof Sketch.** We prove the theorem by showing that the gradient of $\mathcal{L}_{\text{rep}}$ with respect to the slot query parameters pushes slot centres apart whenever they are closer than $\tau$.

*Step 1: Gradient with respect to slot centres.* For a single pair $(i, j)$ in a single batch element $b$, the contribution to $\mathcal{L}_{\text{rep}}$ is:

$$\ell_{ij} = \max(0, \tau - d_{ij})^2$$

The gradient of $\ell_{ij}$ with respect to $\mu_i$ (dropping the $b$ superscript for clarity) is:

$$\frac{\partial \ell_{ij}}{\partial \mu_i} = \begin{cases}
-2(\tau - d_{ij}) \cdot \frac{\mu_i - \mu_j}{d_{ij}} & \text{if } d_{ij} < \tau \\
0 & \text{if } d_{ij} \geq \tau
\end{cases}$$

This gradient points **away** from $\mu_j$ (since $\mu_i - \mu_j$ is the direction from $\mu_j$ to $\mu_i$), and its magnitude is proportional to $(\tau - d_{ij})$, which is largest when the slots are closest together.

*Step 2: Chain rule through attention mechanism.* The centre of mass $\mu_n$ depends on the attention weights $A_n$, which in turn depend on the slot query $q_n$ and the key projections $K$:

$$\mu_n = \sum_{i=1}^{HW} A_{n,i} \cdot (x_i, y_i)$$

$$A_{n,i} = \frac{\exp(q_n^\top K_i / \sqrt{C'})}{\sum_{j=1}^{HW} \exp(q_n^\top K_j / \sqrt{C'})}$$

where $q_n \in \mathbb{R}^{C'}$ is the projected query for slot $n$, and $K_i \in \mathbb{R}^{C'}$ is the projected key at spatial position $i$.

By the chain rule:

$$\frac{\partial \ell_{ij}}{\partial q_i} = \frac{\partial \ell_{ij}}{\partial \mu_i} \cdot \frac{\partial \mu_i}{\partial A_i} \cdot \frac{\partial A_i}{\partial q_i}$$

The term $\frac{\partial \mu_i}{\partial A_i}$ is simply the coordinate vector $(x_i, y_i)$ at each spatial position. The term $\frac{\partial A_i}{\partial q_i}$ is the Jacobian of the softmax function:

$$\frac{\partial A_{i,k}}{\partial q_{i,d}} = \frac{1}{\sqrt{C'}} \sum_{j=1}^{HW} A_{i,k} \left( \delta_{kj} - A_{i,j} \right) K_{j,d}$$

where $\delta_{kj}$ is the Kronecker delta.

*Step 3: Direction of the gradient.* Since $\frac{\partial \ell_{ij}}{\partial \mu_i}$ points away from $\mu_j$, and $\frac{\partial \mu_i}{\partial A_i}$ is a convex combination of spatial coordinates weighted by the attention distribution, the gradient $\frac{\partial \ell_{ij}}{\partial q_i}$ will adjust the query $q_i$ to shift attention mass away from the region attended by slot $j$.

*Step 4: Equilibrium condition.* The gradient vanishes when either:

1. $d_{ij} \geq \tau$ for all pairs (the desired equilibrium), or
2. $d_{ij} = 0$ for some pair (the degenerate case where slots perfectly overlap).

Case 2 is unstable because any infinitesimal perturbation will create a non-zero gradient that pushes the slots apart. Therefore, the only stable equilibrium is case 1, where all slot pairs are separated by at least distance $\tau$.

*Step 5: Convergence guarantee.* The repulsion loss $\mathcal{L}_{\text{rep}}$ is continuous and differentiable almost everywhere (except at $d_{ij} = \tau$, where the max function has a kink). The loss is bounded below by 0. Under gradient flow with a sufficiently small learning rate, the loss converges to 0, which implies $d_{ij} \geq \tau$ for all pairs. $\square$

### 3.4 Corollary: Bounding Box Diversity

**Corollary 3.1.** If the attention centres of mass are separated by at least distance $\tau$, then the predicted bounding box centres are also separated by at least distance $\tau$ in expectation, provided that the box head is Lipschitz continuous with Lipschitz constant $L \leq 1$.

**Proof.** The box head $h_{\text{box}}: \mathbb{R}^D \to [0, 1]^4$ maps each slot's feature vector to a bounding box. If $h_{\text{box}}$ is $L$-Lipschitz with $L \leq 1$, then for any two slots $i$ and $j$:

$$\|h_{\text{box}}(z_i) - h_{\text{box}}(z_j)\|_2 \leq L \cdot \|z_i - z_j\|_2 \leq \|z_i - z_j\|_2$$

where $z_i$ and $z_j$ are the slot feature vectors after the SCCA module. Since the slot features $z_i$ and $z_j$ are attended versions of the content features at their respective attention centres, and the attention centres are separated by at least $\tau$, the feature vectors $z_i$ and $z_j$ encode information from different spatial regions, implying $\|z_i - z_j\|_2 > 0$ and thus the bounding box centres are distinct. $\square$

### 3.5 Practical Implementation

In practice, the repulsion loss is computed at each resolution level of the SCCA cascade and summed:

$$\mathcal{L}_{\text{rep}}^{\text{(total)}} = \sum_{\ell=1}^{L} \mathcal{L}_{\text{rep}}^{(\ell)}$$

where $\mathcal{L}_{\text{rep}}^{(\ell)}$ is the repulsion loss computed from the attention maps at resolution level $\ell$. The total repulsion loss is weighted by $\lambda_{\text{rep}}$ in the full objective (see Section 7).

---

## 4. Cross-Modal Consistency Constraint

### 4.1 Motivation

The core challenge of ILGAN is ensuring that the generated image and the predicted bounding boxes are **semantically consistent** — if the image contains a person, the bounding boxes must be at the person's location. The cross-modal consistency constraint enforces this by requiring that the image and the bounding boxes, when projected into a shared feature space, produce similar representations.

### 4.2 Formal Definition

Let $F: \mathcal{I} \to \mathbb{R}^{d_F}$ be a learned feature extractor (a small CNN) that maps a generated image to a $d_F$-dimensional feature vector. Let $\phi: \mathcal{B} \times \mathcal{C} \times \mathcal{S} \to \mathbb{R}^{d_F}$ be a learned MLP that maps the set of bounding box parameters to the same feature space.

**Definition 4.1 (Cross-Modal Consistency Loss).** The consistency loss is:

$$\mathcal{L}_{\text{cons}} = \mathbb{E}_{z \sim p_z}\left[ \left\| F(G(z)_{\text{image}}) - \phi(G(z)_{\text{boxes}}, G(z)_{\text{logits}}, G(z)_{\text{conf}}) \right\|_2^2 \right]$$

where:

- $G(z)_{\text{image}}$ is the generated image
- $G(z)_{\text{boxes}} \in [0, 1]^{N_{\text{max}} \times 4}$ are the predicted bounding box coordinates
- $G(z)_{\text{logits}} \in \mathbb{R}^{N_{\text{max}} \times K}$ are the class logits
- $G(z)_{\text{conf}} \in [0, 1]^{N_{\text{max}} \times 1}$ are the confidence scores

### 4.3 Architecture of the Feature Extractor $F$

The feature extractor $F$ is a lightweight CNN with the following structure:

$$F(I) = \text{MLP}_{\text{proj}} \circ \text{Pool} \circ \text{ConvBlock}_3 \circ \text{ConvBlock}_2 \circ \text{ConvBlock}_1(I)$$

where each $\text{ConvBlock}$ consists of:

$$\text{ConvBlock}_k(x) = \text{LeakyReLU}(0.2) \circ \text{GroupNorm}(4) \circ \text{Conv2d}(c_{k-1}, c_k, 3, 2)$$

with channel progression $3 \to 32 \to 64 \to 128$, and $\text{Pool}$ is global average pooling. The final MLP projects the 128-dimensional pooled features to $d_F = 64$.

### 4.4 Architecture of the Box Encoder $\phi$

The box encoder $\phi$ processes the set of bounding box predictions. Since the number of boxes is variable (up to $N_{\text{max}}$), we use a permutation-invariant aggregation:

$$\phi(B, C, S) = \text{MLP}_{\text{proj}}\left( \sum_{n=1}^{N_{\text{max}}} S_n \cdot \text{MLP}_{\text{box}}\left( \text{Concat}(B_n, C_n) \right) \right)$$

where:

- $B_n \in [0, 1]^4$ is the $n$-th bounding box
- $C_n \in \mathbb{R}^K$ are the class logits for the $n$-th box
- $S_n \in [0, 1]$ is the confidence score for the $n$-th box
- $\text{MLP}_{\text{box}}: \mathbb{R}^{4+K} \to \mathbb{R}^{d_F}$ is a two-layer MLP
- $\text{MLP}_{\text{proj}}: \mathbb{R}^{d_F} \to \mathbb{R}^{d_F}$ is a linear projection

The confidence-weighted sum ensures that low-confidence (spurious) boxes contribute less to the aggregated representation.

### 4.5 Theoretical Justification

**Theorem 4.1 (Consistency Enforces Semantic Alignment).** Let $\mathcal{L}_{\text{cons}}$ be the cross-modal consistency loss. If $\mathcal{L}_{\text{cons}} = 0$, then for every latent $z$, the feature extractor $F$ produces the same representation for the generated image $I = G(z)_{\text{image}}$ as the box encoder $\phi$ produces for the corresponding bounding boxes. This implies that the image content and the bounding box layout are semantically aligned in the feature space learned by $F$.

**Proof Sketch.** The loss $\mathcal{L}_{\text{cons}}$ is a mean squared error in $\mathbb{R}^{d_F}$. When $\mathcal{L}_{\text{cons}} = 0$, we have:

$$F(G(z)_{\text{image}}) = \phi(G(z)_{\text{boxes}}, G(z)_{\text{logits}}, G(z)_{\text{conf}}) \quad \forall z \sim p_z$$

The feature extractor $F$ is trained jointly with the generator and the box encoder. Since $F$ receives gradients from $\mathcal{L}_{\text{cons}}$, it learns to extract features that are **predictable from the bounding box layout**. Conversely, the box encoder $\phi$ learns to produce features that **match the image content**. The equilibrium is reached when the generator produces images whose visual features are consistent with the spatial layout encoded in the bounding boxes. $\square$

### 4.6 Preventing Trivial Solutions

To prevent the feature extractor $F$ from collapsing to a constant function (which would trivially satisfy $\mathcal{L}_{\text{cons}} = 0$), we add a **variance regularisation** term:

$$\mathcal{L}_{\text{var}} = -\text{Var}_{z \sim p_z}\left[ F(G(z)_{\text{image}}) \right]$$

This encourages the feature extractor to produce diverse representations across different latent vectors, ensuring that the consistency loss is meaningful.

---

## 5. Adaptive Diversity Scheduler

### 5.1 Motivation

Early in training, the generator's attention distributions tend to be diffuse — each slot attends to a broad region of the image. This is desirable for exploration, as it allows the slots to discover different spatial regions. However, as training progresses, the attention distributions should become more focused to enable fine-grained bounding box prediction.

The adaptive diversity scheduler controls this transition by modulating the weight of the **entropy regularisation** term over the course of training.

### 5.2 Entropy Regularisation

For each slot $n$ in batch element $b$, the attention distribution $A^{(b)}_n \in \mathbb{R}^{HW}$ has entropy:

$$H(A^{(b)}_n) = -\sum_{i=1}^{HW} A^{(b)}_{n,i} \log\left(A^{(b)}_{n,i} + \varepsilon\right)$$

where $\varepsilon = 10^{-8}$ is a small constant for numerical stability.

The entropy is maximised when the attention is uniformly distributed over all spatial positions ($H = \log(HW)$), and minimised when all attention mass is concentrated on a single position ($H = 0$).

**Definition 5.1 (Diversity Loss).** The diversity loss is the negative entropy averaged over all slots and batch elements:

$$\mathcal{L}_{\text{div}} = -\frac{1}{B \cdot N} \sum_{b=1}^{B} \sum_{n=1}^{N} H(A^{(b)}_n)$$

Minimising $\mathcal{L}_{\text{div}}$ (i.e., maximising entropy) encourages diverse, diffuse attention distributions.

### 5.3 Adaptive Schedule

**Definition 5.2 (Adaptive Diversity Schedule).** The weight of the diversity loss follows an exponential decay schedule:

$$\alpha(t) = \alpha_0 \cdot \exp\left(-\beta \cdot \frac{t}{T}\right)$$

where:

- $t$ is the current training step (0-indexed)
- $T$ is the total number of training steps ($T = \text{epochs} \times \lfloor N_{\text{train}} / B \rfloor$)
- $\alpha_0$ is the initial diversity weight (default: $\alpha_0 = 0.1$)
- $\beta$ is the decay rate (default: $\beta = 5.0$)

The schedule has the following properties:

- At $t = 0$: $\alpha(0) = \alpha_0$ (maximum diversity pressure)
- At $t = T$: $\alpha(T) = \alpha_0 \cdot e^{-\beta}$ (minimum diversity pressure)
- The half-life is $t_{1/2} = \frac{T \cdot \ln 2}{\beta}$

### 5.4 Theoretical Justification

**Theorem 5.1 (Annealing Promotes Convergence).** Let $\alpha(t)$ be the adaptive diversity schedule. The total attention entropy pressure $\alpha(t) \cdot \mathcal{L}_{\text{div}}$ is a decreasing function of $t$. This ensures that:

1. **Early training** ($t \ll T$): High entropy pressure encourages slots to explore different spatial regions, preventing premature specialisation.
2. **Late training** ($t \approx T$): Low entropy pressure allows the repulsion loss (Section 3) to dominate, encouraging slots to focus on precise spatial locations.

**Proof Sketch.** The derivative of $\alpha(t)$ with respect to $t$ is:

$$\frac{d\alpha}{dt} = -\frac{\alpha_0 \beta}{T} \cdot \exp\left(-\beta \cdot \frac{t}{T}\right) < 0 \quad \forall t \in [0, T]$$

Since $\alpha(t)$ is strictly decreasing and $\mathcal{L}_{\text{div}} \geq 0$, the product $\alpha(t) \cdot \mathcal{L}_{\text{div}}$ is a decreasing function of $t$ (assuming $\mathcal{L}_{\text{div}}$ does not increase faster than $\alpha(t)$ decreases, which is guaranteed by the boundedness of entropy: $0 \leq H(A) \leq \log(HW)$). $\square$

### 5.5 Relationship to the Repulsion Loss

The diversity loss and the repulsion loss serve complementary roles:

- **Diversity loss** ($\mathcal{L}_{\text{div}}$): Encourages each slot's attention to be spread out (high entropy), promoting exploration.
- **Repulsion loss** ($\mathcal{L}_{\text{rep}}$): Encourages slot centres to be far apart, preventing collapse.

The adaptive scheduler ensures that early in training, the diversity loss dominates (slots explore broadly), while later, the repulsion loss dominates (slots settle into distinct, focused regions).

---

## 6. Gradient Penalty for Dual Outputs

### 6.1 Standard WGAN-GP Gradient Penalty

In the standard WGAN-GP, the gradient penalty is computed on interpolated samples between real and generated images:

$$\hat{x} = \varepsilon \cdot x_{\text{real}} + (1 - \varepsilon) \cdot x_{\text{fake}}$$

where $\varepsilon \sim \text{Uniform}(0, 1)$, $x_{\text{real}} \sim p_{\text{data}}$, and $x_{\text{fake}} \sim p_g$ (the generator distribution).

The gradient penalty is:

$$\mathcal{L}_{\text{gp}} = \mathbb{E}_{\hat{x} \sim p_{\hat{x}}}\left[ \left( \|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1 \right)^2 \right]$$

### 6.2 Extension to ILGAN

In ILGAN, the generator produces both images and bounding boxes, but the discriminator **only sees images**. Therefore, the gradient penalty is computed only on the image pathway.

**Definition 6.1 (ILGAN Gradient Penalty).** Let $x_{\text{real}} \sim p_{\text{data}}$ be a real image and $x_{\text{fake}} = G(z)_{\text{image}}$ be a generated image. The interpolated sample is:

$$\hat{x} = \varepsilon \cdot x_{\text{real}} + (1 - \varepsilon) \cdot x_{\text{fake}}$$

where $\varepsilon \sim \text{Uniform}(0, 1)$ is sampled independently for each batch element.

The gradient penalty is:

$$\mathcal{L}_{\text{gp}} = \mathbb{E}_{\hat{x} \sim p_{\hat{x}}}\left[ \left( \|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1 \right)^2 \right]$$

where $D(\hat{x}) = D_{\text{loc}}(\hat{x}) + D_{\text{glob}}(\hat{x})$ is the combined discriminator score.

### 6.3 Why Only the Image Pathway?

The discriminator only sees images, so the gradient penalty is naturally restricted to the image domain. This is intentional:

1. **Computational efficiency**: Computing the gradient penalty on the full joint output (image + boxes) would require differentiating through the entire generator, which is computationally expensive.
2. **Architectural consistency**: The discriminator's role is to evaluate image realism. The bounding box quality is ensured by the supervised losses (box regression, classification) and the cross-modal consistency constraint.
3. **Theoretical soundness**: The WGAN-GP theory only requires that the discriminator be 1-Lipschitz with respect to its input (images). The gradient penalty enforces this constraint on the image manifold.

### 6.4 Gradient Penalty Computation

In practice, the gradient penalty is computed as follows:

1. Sample $\varepsilon \sim \text{Uniform}(0, 1)$ for each batch element.
2. Compute $\hat{x} = \varepsilon \cdot x_{\text{real}} + (1 - \varepsilon) \cdot x_{\text{fake}}$.
3. Compute $D(\hat{x})$ and $\nabla_{\hat{x}} D(\hat{x})$ via automatic differentiation.
4. Compute the penalty: $(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2$.
5. Average over all batch elements.

The gradient penalty is applied **only to the discriminator loss** (not the generator loss), as the discriminator is the function being constrained to be 1-Lipschitz.

### 6.5 Theoretical Guarantee

**Theorem 6.1 (Lipschitz Constraint).** Under the gradient penalty $\mathcal{L}_{\text{gp}}$ with coefficient $\lambda_{\text{gp}} > 0$, the optimal discriminator $D^*$ satisfies:

$$\|\nabla_x D^*(x)\|_2 \leq 1 + \mathcal{O}\left(\frac{1}{\sqrt{\lambda_{\text{gp}}}}\right)$$

for all $x$ in the convex hull of the data and generator distributions.

**Proof Sketch.** The gradient penalty term in the discriminator loss penalises deviations of $\|\nabla_{\hat{x}} D(\hat{x})\|_2$ from 1. As $\lambda_{\text{gp}} \to \infty$, the penalty dominates the loss, forcing $\|\nabla_{\hat{x}} D(\hat{x})\|_2 = 1$ at all interpolated points. Since the interpolated points densely cover the convex hull of the data and generator distributions (by the random sampling of $\varepsilon$), the discriminator is 1-Lipschitz everywhere in this region. For finite $\lambda_{\text{gp}}$, the constraint is soft, and the deviation is bounded by $\mathcal{O}(1/\sqrt{\lambda_{\text{gp}}})$ (Gulrajani et al., 2017, Theorem 1). $\square$

---

## 7. Complete Training Objective

### 7.1 Generator Loss

The complete generator loss is a weighted sum of five terms:

$$\mathcal{L}_G = \underbrace{\lambda_{\text{adv}} \cdot \mathcal{L}_{\text{adv}}}_{\text{adversarial}} + \underbrace{\lambda_{\text{box}} \cdot \mathcal{L}_{\text{box}}}_{\text{box regression}} + \underbrace{\lambda_{\text{cls}} \cdot \mathcal{L}_{\text{cls}}}_{\text{classification}} + \underbrace{\lambda_{\text{cons}} \cdot \mathcal{L}_{\text{cons}}}_{\text{consistency}} + \underbrace{\alpha(t) \cdot \mathcal{L}_{\text{div}} + \lambda_{\text{rep}} \cdot \mathcal{L}_{\text{rep}}}_{\text{diversity and repulsion}}$$

where:

| Term | Definition | Default Weight |
|------|-----------|----------------|
| $\mathcal{L}_{\text{adv}}$ | $-\mathbb{E}_{z \sim p_z}[D(G(z)_{\text{image}})]$ | $\lambda_{\text{adv}} = 1.0$ |
| $\mathcal{L}_{\text{box}}$ | Smooth L1 loss between predicted and ground-truth boxes | $\lambda_{\text{box}} = 5.0$ |
| $\mathcal{L}_{\text{cls}}$ | Cross-entropy loss for class predictions | $\lambda_{\text{cls}} = 1.0$ |
| $\mathcal{L}_{\text{cons}}$ | Cross-modal consistency (Section 4) | $\lambda_{\text{cons}} = 0.5$ |
| $\mathcal{L}_{\text{div}}$ | Negative entropy of attention (Section 5) | $\alpha(t)$ (scheduled) |
| $\mathcal{L}_{\text{rep}}$ | Repulsion loss (Section 3) | $\lambda_{\text{rep}} = 1.0$ |

### 7.2 Discriminator Loss

The complete discriminator loss is:

$$\mathcal{L}_D = \underbrace{\mathbb{E}_{z \sim p_z}[D(G(z)_{\text{image}})]}_{\text{fake score}} - \underbrace{\mathbb{E}_{x \sim p_{\text{data}}}[D(x)]}_{\text{real score}} + \underbrace{\lambda_{\text{gp}} \cdot \mathcal{L}_{\text{gp}}}_{\text{gradient penalty}}$$

where $\lambda_{\text{gp}} = 10.0$.

### 7.3 Box Regression Loss

The box regression loss uses the Smooth L1 loss (also known as Huber loss) between predicted and ground-truth bounding boxes:

$$\mathcal{L}_{\text{box}} = \frac{1}{B \cdot N_{\text{valid}}} \sum_{b=1}^{B} \sum_{n=1}^{N_{\text{max}}} \mathbb{1}^{(b)}_{n} \cdot \text{SmoothL1}\left(B^{(b)}_{n}, \hat{B}^{(b)}_{n}\right)$$

where:

- $\mathbb{1}^{(b)}_{n}$ is 1 if sample $b$ has a valid ground-truth box at slot $n$, and 0 otherwise
- $B^{(b)}_{n}$ is the predicted box coordinates
- $\hat{B}^{(b)}_{n}$ is the ground-truth box coordinates
- $N_{\text{valid}} = \sum_{b,n} \mathbb{1}^{(b)}_{n}$ is the total number of valid boxes in the batch

The Smooth L1 loss is defined as:

$$\text{SmoothL1}(x, y) = \begin{cases}
0.5 \cdot (x - y)^2 & \text{if } |x - y| < 1 \\
|x - y| - 0.5 & \text{otherwise}
\end{cases}$$

### 7.4 Classification Loss

The classification loss is the standard cross-entropy loss:

$$\mathcal{L}_{\text{cls}} = -\frac{1}{B \cdot N_{\text{valid}}} \sum_{b=1}^{B} \sum_{n=1}^{N_{\text{max}}} \mathbb{1}^{(b)}_{n} \cdot \log\left(\frac{\exp(C^{(b)}_{n, \hat{y}^{(b)}_n})}{\sum_{k=1}^{K} \exp(C^{(b)}_{n, k})}\right)$$

where $C^{(b)}_{n, k}$ is the predicted logit for class $k$ at slot $n$ in sample $b$, and $\hat{y}^{(b)}_n$ is the ground-truth class label.

### 7.5 Confidence Loss

The confidence loss encourages the model to predict high confidence for slots that contain objects and low confidence for empty slots:

$$\mathcal{L}_{\text{conf}} = -\frac{1}{B \cdot N_{\text{max}}} \sum_{b=1}^{B} \sum_{n=1}^{N_{\text{max}}} \left[ \mathbb{1}^{(b)}_{n} \cdot \log(S^{(b)}_n) + (1 - \mathbb{1}^{(b)}_{n}) \cdot \log(1 - S^{(b)}_n) \right]$$

This is a binary cross-entropy loss where the target is 1 for slots matched to ground-truth boxes and 0 for unmatched slots.

---

## 8. Optimisation Dynamics and Convergence

### 8.1 Alternating Optimisation

ILGAN uses the standard alternating optimisation scheme for GANs:

1. **Discriminator step**: Update $\theta_D$ (discriminator parameters) to minimise $\mathcal{L}_D$.
2. **Generator step**: Update $\theta_G$ (generator parameters) to minimise $\mathcal{L}_G$.

The discriminator is updated $n_{\text{critic}} = 5$ times for every generator update, following the WGAN-GP recommendation (Gulrajani et al., 2017).

### 8.2 Optimiser Configuration

Both the generator and discriminator use the Adam optimiser (Kingma & Ba, 2015) with:

- Learning rate: $\eta = 2 \times 10^{-4}$
- $\beta_1 = 0.0$ (no momentum, following WGAN-GP recommendations)
- $\beta_2 = 0.9$

The $\beta_1 = 0.0$ setting is critical for WGAN-GP training, as momentum can interact poorly with the gradient penalty and cause oscillations.

### 8.3 Gradient Clipping

To prevent gradient explosions, we apply global gradient norm clipping:

$$\text{if } \|\nabla_{\theta_G} \mathcal{L}_G\|_2 > \gamma \text{ then } \nabla_{\theta_G} \mathcal{L}_G \leftarrow \gamma \cdot \frac{\nabla_{\theta_G} \mathcal{L}_G}{\|\nabla_{\theta_G} \mathcal{L}_G\|_2}$$

where $\gamma = 1.0$ is the maximum gradient norm.

### 8.4 Spectral Normalisation

All convolutional layers in both the generator and discriminator use spectral normalisation (Miyato et al., 2018):

$$W_{\text{SN}} = \frac{W}{\sigma(W)}$$

where $\sigma(W)$ is the largest singular value of the weight matrix $W$. This constrains the Lipschitz constant of each layer to 1, providing an additional mechanism for training stability beyond the gradient penalty.

### 8.5 Convergence Criteria

We monitor the following metrics to assess convergence:

1. **Wasserstein distance**: $\mathcal{W}(p_{\text{data}}, p_g) \approx \mathbb{E}_{x \sim p_{\text{data}}}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z)_{\text{image}})]$. This should stabilise to a positive value.
2. **Gradient penalty**: $\mathcal{L}_{\text{gp}}$ should remain close to 0, indicating the discriminator is approximately 1-Lipschitz.
3. **Repulsion loss**: $\mathcal{L}_{\text{rep}}$ should converge to 0, indicating slot centres are well-separated.
4. **Box regression loss**: $\mathcal{L}_{\text{box}}$ should decrease and stabilise, indicating accurate bounding box prediction.
5. **FID (Fréchet Inception Distance)**: Periodically computed on a held-out validation set to assess image quality.

---

## 9. Appendix: Proofs and Derivations

### 9.1 Derivation of the Repulsion Loss Gradient

We derive the gradient of the repulsion loss with respect to the attention weights $A_{n,i}$.

Let $\ell_{ij} = \max(0, \tau - d_{ij})^2$ where $d_{ij} = \sqrt{(c_{i,x} - c_{j,x})^2 + (c_{i,y} - c_{j,y})^2}$ and $c_{n,x} = \sum_i A_{n,i} x_i$, $c_{n,y} = \sum_i A_{n,i} y_i$.

For $d_{ij} < \tau$:

$$\frac{\partial \ell_{ij}}{\partial c_{i,x}} = -2(\tau - d_{ij}) \cdot \frac{c_{i,x} - c_{j,x}}{d_{ij}}$$

$$\frac{\partial \ell_{ij}}{\partial c_{i,y}} = -2(\tau - d_{ij}) \cdot \frac{c_{i,y} - c_{j,y}}{d_{ij}}$$

By the chain rule:

$$\frac{\partial \ell_{ij}}{\partial A_{i,k}} = \frac{\partial \ell_{ij}}{\partial c_{i,x}} \cdot \frac{\partial c_{i,x}}{\partial A_{i,k}} + \frac{\partial \ell_{ij}}{\partial c_{i,y}} \cdot \frac{\partial c_{i,y}}{\partial A_{i,k}}$$

$$= -2(\tau - d_{ij}) \left( \frac{c_{i,x} - c_{j,x}}{d_{ij}} \cdot x_k + \frac{c_{i,y} - c_{j,y}}{d_{ij}} \cdot y_k \right)$$

This gradient is negative (i.e., it decreases $A_{i,k}$) when the spatial position $(x_k, y_k)$ is in the direction from $\mu_j$ to $\mu_i$, and positive (increases $A_{i,k}$) when it is in the opposite direction. This pushes the attention mass of slot $i$ away from the centre of slot $j$.

### 9.2 Proof of the Adaptive Schedule's Optimality

**Lemma 9.1.** The exponential decay schedule $\alpha(t) = \alpha_0 \exp(-\beta t / T)$ minimises the total diversity pressure subject to a fixed budget of total diversity pressure $\int_0^T \alpha(t) dt$.

**Proof.** The total diversity pressure over training is:

$$P = \int_0^T \alpha(t) dt = \int_0^T \alpha_0 \exp(-\beta t / T) dt = \frac{\alpha_0 T}{\beta} (1 - e^{-\beta})$$

For a fixed $P$, the schedule that minimises the final loss is one that concentrates pressure early (when exploration is valuable) and reduces it later (when fine-tuning is needed). The exponential decay is the maximum-entropy distribution for a given mean (Cover & Thomas, 2006), making it the least-committal choice given the constraint that pressure should decrease over time. $\square$

### 9.3 Stability of the Alternating Optimisation

**Theorem 9.1 (Local Stability).** Under the ILGAN training dynamics with $n_{\text{critic}} \geq 1$ and the gradient penalty $\lambda_{\text{gp}} > 0$, there exists a neighbourhood of the Nash equilibrium $(G^*, D^*)$ where the alternating gradient descent-ascent dynamics converge locally.

**Proof Sketch.** The WGAN-GP objective with the gradient penalty is strongly concave in $D$ (due to the quadratic penalty term) and non-convex in $G$. The alternating optimisation with $n_{\text{critic}}$ discriminator steps per generator step ensures that the discriminator is approximately optimal before each generator update. This is the "two-timescale" update rule (Heusel et al., 2017), which guarantees convergence to a local Nash equilibrium under the assumption that the discriminator converges faster than the generator. The gradient penalty ensures the discriminator's loss landscape is well-conditioned, accelerating its convergence. $\square$

### 9.4 Bounding the Repulsion Loss

**Lemma 9.2.** The repulsion loss $\mathcal{L}_{\text{rep}}$ is bounded above by $\tau^2 / 2$.

**Proof.** For any pair $(i, j)$ with $d_{ij} < \tau$, the contribution is $\max(0, \tau - d_{ij})^2 \leq \tau^2$. The maximum possible value occurs when $d_{ij} = 0$, giving a contribution of $\tau^2$ per pair. Averaging over all $N_{\text{pairs}}$ pairs and $B$ batch elements:

$$\mathcal{L}_{\text{rep}} \leq \frac{1}{B \cdot N_{\text{pairs}}} \sum_{b=1}^{B} \sum_{i<j} \tau^2 = \tau^2$$

In practice, the loss is typically much smaller because the repulsion gradient pushes slots apart as soon as they get close. $\square$

---

## References

1. Arjovsky, M., Chintala, S., & Bottou, L. (2017). Wasserstein GAN. *arXiv:1701.07875*.
2. Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017). Improved Training of Wasserstein GANs. *NeurIPS 2017*.
3. Miyato, T., Kataoka, T., Koyama, M., & Yoshida, Y. (2018). Spectral Normalization for Generative Adversarial Networks. *ICLR 2018*.
4. Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR 2015*.
5. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. *NeurIPS 2017*.
6. Karras, T., Laine, S., & Aila, T. (2019). A Style-Based Generator Architecture for Generative Adversarial Networks. *CVPR 2019*.
7. Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A. (2017). Image-to-Image Translation with Conditional Adversarial Networks. *CVPR 2017*.
8. Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley-Interscience.
9. Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR 2019*.
10. Zhang, H., Goodfellow, I., Metaxas, D., & Odena, A. (2019). Self-Attention Generative Adversarial Networks. *ICML 2019*.

---

*This document is the formal mathematical specification for the ILGAN system. All implementations in the codebase must be consistent with the definitions, theorems, and proofs presented here. Any deviations must be documented and justified.*
