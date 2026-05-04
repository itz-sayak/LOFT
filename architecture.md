```markdown
# Semantics Know the Shape: Latent-Conditioned Flow Matching for 3D Point Cloud Denoising

## LOFT — Latent-Guided Optimal Flow Transport

### How to Use This Document for draw.io

Each section below describes one block or group of blocks in the architecture diagram. The data tensor shapes are written in square brackets, e.g. [B, 3, 2048], and represent the arrow labels connecting blocks. Each subsection maps directly to one rectangular block in draw.io, and the "Input → Output" lines tell you how to draw the connecting arrows.

---

## 1. Top-Level System Block

The full system is called **LatentOTFlowBridge**. It takes as input a batch of noisy 3D point patches of shape **[B, 3, 2048]** — where B is the batch size, 3 is the XYZ channel dimension, and 2048 is the number of points per patch. The system outputs a corresponding batch of clean 3D point patches of the same shape **[B, 3, 2048]**. During training, the system also accepts a paired clean patch [B, 3, 2048] and computes a scalar training loss. During inference, only the noisy input is required and the system runs a 10-step Euler ODE solver to produce the denoised output. This top-level block contains three internal sub-systems: the **Frozen Semantic Autoencoder**, the **Trainable FreqEncodingTransformer**, and the **PVCNN2Unet Denoising Backbone**. These three blocks are connected in a sequential conditioning pipeline described in the sections below.

> **draw.io note:** Draw this as a large outer rectangle labelled `LatentOTFlowBridge` with three inner rectangles arranged left to right: `SemanticAE → FreqEncodingTransformer → PVCNN2Unet`. Above the outer block, draw an incoming arrow labelled `"Noisy patch [B, 3, 2048]"`. Below it, draw an outgoing arrow labelled `"Clean patch [B, 3, 2048]"`.

---

## 2. Optimal Transport Coupling Block *(Training Only)*

Before any neural network processing occurs during training, the system performs an **Optimal Transport Coupling** step that re-orders the noisy and clean patches in each batch to minimize total pointwise transport cost. This block takes as input the raw noisy batch [B, 3, 2048] and the raw clean batch [B, 3, 2048] as two separate incoming arrows. Internally, it computes a per-sample N×N squared Euclidean distance matrix on GPU (where N=2048), transfers it to CPU, and solves the assignment problem using the **Hungarian algorithm** in a parallel thread pool. The output is a reordered noisy batch [B, 3, 2048] and reordered clean batch [B, 3, 2048] where each noisy patch is paired with the clean patch that minimises transport cost. These two reordered tensors flow to the Interpolation block.

> **draw.io note:** Draw this as a rectangle labelled `"OT Coupling (Hungarian)"` with two input arrows from the left (`"Noisy [B,3,2048]"` and `"Clean [B,3,2048]"`) and two output arrows to the right (`"x0 reordered [B,3,2048]"` and `"x1 reordered [B,3,2048]"`).

---

## 3. Interpolation Block *(Training Only)*

The **Interpolation block** takes the OT-coupled noisy points x0 [B, 3, 2048], the clean points x1 [B, 3, 2048], and a randomly sampled scalar timestep t drawn from a Uniform(0,1) distribution. From these inputs it computes the linearly interpolated intermediate state x_t using the formula:

```
x_t = (1−t)·x0 + t·x1 + σ_min·ε
```

where ε is Gaussian noise and σ_min = 1×10⁻⁴ is a small noise floor that prevents the interpolation from collapsing to a degenerate distribution. The output is a single interpolated point cloud **x_t [B, 3, 2048]** and the scalar timestep **t [B]**, both of which are passed to the conditioning pipeline and the backbone.

> **draw.io note:** Draw this as a rectangle labelled `"Linear Interpolation + noise floor σ_min=1e-4"` with three input arrows (x0, x1, t ~ U(0,1)) and one output arrow labelled `"x_t [B, 3, 2048]"`. Draw a separate small circle or diamond labelled `"t ~ U(0,1)"` with an arrow feeding both this block and the FreqEncodingTransformer.

---

## 4. SemanticAutoencoder Block *(Frozen)*

The **SemanticAutoencoder** is a PointNet++ based autoencoder that was trained separately on clean point cloud geometry and is kept entirely frozen during LOFT training — none of its **5,985,653 parameters** receive gradients. Its role is to act as a geometry prior: given a noisy point patch as input, it encodes the rough shape into a compact set of semantic latent tokens that describe what the underlying clean shape should look like.

Internally, the encoder uses three **Set Abstraction (SA) layers** that progressively downsample the input point cloud from 2048 points to 1024, then to 256, then to 64 representative points. Each SA layer uses ball-query grouping and PointNet-style local feature aggregation. After the SA layers, the encoder applies an **OffsetAttention** module (a self-attention variant that uses feature offsets rather than raw features), a **CrossAttention** module for global context aggregation, and **Squeeze-and-Excitation (SE)** channel-weighting blocks. The final output of the encoder is a set of **64 latent tokens** each of dimension 512, forming a tensor of shape **[B, 64, 512]**.

The input to this block is the noisy point patch **x0 [B, 3, 2048]** (not the interpolated x_t — the AE always encodes the original noisy input). The output is the **AE latent tensor [B, 64, 512]**. This tensor flows to the FreqEncodingTransformer.

> **draw.io note:** Draw this as a rectangle labelled `"SemanticAutoencoder (Frozen, 5.99M params)"` with a `"FROZEN"` badge. Inside, draw three smaller rectangles stacked: `"SA 2048→1024"`, `"SA 1024→256"`, `"SA 256→64"`, followed by `"OffsetAttn + CrossAttn + SE"`. The input arrow is `"x0 [B, 3, 2048]"` and the output arrow is `"AE latent tokens [B, 64, 512]"`.

---

## 5. FreqEncodingTransformer Block *(Trainable)*

The **FreqEncodingTransformer** is a transformer-based module with **17,389,184 trainable parameters** that refines the AE latent tokens by making them aware of the current position along the flow trajectory (i.e., the timestep t). Raw AE tokens carry only geometric shape information with no knowledge of where the denoising process currently is; this module injects that temporal context.

The module first applies a **FourierFreqEncoding** layer, which computes a learnable Fourier positional encoding over the 64 token positions using **6 frequency bands**, producing a positional embedding that is added to the AE token sequence. It then passes the token sequence through **4 stacked Transformer Encoder layers**. Each such layer consists of:

- Multi-head self-attention with **8 heads**
- A feed-forward network with expansion factor 4 (hidden dim = 4×512 = 2048)
- Layer normalisation
- Residual connections
- Dropout of 0.1

Crucially, each of the 4 transformer layers also contains a **TimestepFiLM** (Feature-wise Linear Modulation) sub-module: the scalar timestep t is projected through a sinusoidal embedding to a vector, and this vector predicts per-channel scale and shift parameters that modulate the token features after the feed-forward sub-layer. This is how time information permeates the latent token sequence. After the 4 layers, a final **LayerNorm** is applied, and the output tokens have the same shape as the input: **[B, 64, 512]**.

The inputs to this block are the **AE latent tokens [B, 64, 512]** and the **scalar timestep t·1000 [B]** (the timestep is scaled from [0,1] to [0,1000] to match the sinusoidal embedding range). The output is a set of **refined conditioning tokens Z [B, 64, 512]**. These tokens flow into the PVCNN2Unet backbone at five injection points.

> **draw.io note:** Draw this as a rectangle labelled `"FreqEncodingTransformer (Trainable, 17.39M params)"`. Inside, show sequentially: `"FourierFreqEncoding (6 bands, learnable)"` → `"×4 [TransformerEncoderLayer + TimestepFiLM]"` → `"LayerNorm"`. Two input arrows enter: `"AE latent [B,64,512]"` from the left and `"t × 1000 [B]"` from below. One output arrow labelled `"Cond tokens Z [B, 64, 512]"` exits to the right toward the backbone.

---

## 6. PVCNN2Unet Backbone Block

The **PVCNN2Unet** is a PointVoxel-CNN U-Net with **~20.46 million total parameters** (~19.4M excluding write_attn). It takes the interpolated noisy state **x_t [B, 3, 2048]** and the scaled timestep **t·1000 [B]** as primary inputs, and receives the conditioning tokens **Z [B, 64, 512]** at five injection points via **LatentWriteAttention** cross-attention modules. It outputs a **predicted velocity field v̂ [B, 3, 2048]**, which represents the direction and magnitude of movement from x_t toward the clean target x1.

The backbone is a U-Net with an encoder path (Set Abstraction layers), a bottleneck, and a decoder path (Feature Propagation layers). The channel progression through the encoder is **32 → 64 → 128 → 256 → 512**, with voxel resolutions of **32 → 16 → 8 → 8**. There are four SA stages with block counts [1, 2, 1, 1] and ball-query radii [0.1, 0.2, 0.4, 0.8]. The decoder mirrors this with four FP stages with block counts [1, 2, 1, 1], progressively upsampling features back to the full 2048-point resolution.

Every SA and FP block contains an **AdaGN (Adaptive Group Normalization) FiLM** layer that applies the global conditioning signal derived from the mean-pooled latent tokens. This is a coarse global conditioning pathway: the 64 conditioning tokens are averaged to a single **[B, 512]** vector, projected through a 2-layer MLP with SiLU activation, and this vector modulates the group normalisation affine parameters at every resolution level of the backbone.

The bottleneck (deepest SA level) applies a **LinearAttention** module (with 4 heads) over the 512-dimensional features to capture global spatial context before the decoder begins. This is the only full self-attention operation in the backbone; all other layers use convolution and FiLM modulation.

After the four FP decoder stages, the final features **[B, 32, 2048]** pass through a three-layer output MLP: a hidden layer of width 128, dropout, and a final linear projection to dimensionality 3. The output is the **predicted velocity field v̂ [B, 3, 2048]**.

> **draw.io note:** Draw this as a tall U-Net shaped block labelled `"PVCNN2Unet Backbone (~20.46M params)"`. On the left descending side, draw four boxes: `"SA L1 [B,32,N1]"`, `"SA L2 [B,64,N2]"`, `"SA L3 [B,128,N3]"`, `"SA L4 [B,256,N4]"`. At the bottom draw `"Bottleneck [B,512,N4] + LinearAttn"`. On the right ascending side draw four boxes: `"FP L4 [B,256,2048]"`, `"FP L3 [B,128,2048]"`, `"FP L2 [B,64,2048]"`, `"FP L1 [B,32,2048]"`. At the top-right draw `"Output MLP → v̂ [B,3,2048]"`. Draw skip connections as horizontal arrows from the left-side SA boxes to the matching FP boxes.

---

## 7. LatentWriteAttention Block *(×5 instances)*

**LatentWriteAttention** is a cross-attention module that injects the conditioning tokens Z into the backbone's point-level feature maps. There are **five instances** of this module in total: one at the encoder input operating on the initial embedded features **[B, 32, 2048]**, and one at each of the four FP decoder levels operating on features of dimensions 256, 256, 128, and 64 respectively.

Each instance performs the same operation. The spatial point features **F [B, D, N]** (where D is the feature channel dimension at that level and N is the number of points at that resolution) serve as the **query** sequence. The conditioning tokens **Z [B, 64, 512]** serve as both the **key** sequence and the **value** sequence. Before computing attention, a LayerNorm is applied to the query features (normalising over D channels per point) and a separate LayerNorm is applied to the key tokens (normalising over 512 channels per token). This allows the model to work with query and key dimensions that differ (D ≠ 512) via separate learned projection matrices `to_k` and `to_v`.

The attention is computed as standard multi-head dot-product attention with **8 heads**. The head dimension is D/8 at the encoder input (D=32, so head_dim=4) and varies at the FP levels. The attention scores are computed between the N point queries and the M=64 token keys, producing attention weights of shape **[B, h, N, 64]** after softmax. These weights are applied to the value tokens to produce a context output **[B, h, N, head_dim]**, which is reshaped to **[B, N, D]** and projected through a final linear layer. The initialisation of this output projection uses `xavier_uniform` with gain 0.5 and zero bias, which ensures that at the start of training the write_attn contribution is small and the backbone does not experience a conditioning shock. The final output of the module is the residual updated feature map **F + write_attn(F, Z) [B, D, N]**.

> **draw.io note:** Draw each instance as a small rectangle labelled `"LatentWriteAttention"` with two input arrows: one from the backbone feature map F (labelled with the appropriate `[B,D,N]` shape) entering as the Query, and one from the conditioning tokens Z `[B,64,512]` entering as Key/Value. The output arrow returns to the backbone flow labelled `"F + context [B,D,N]"`. Show all five instances with their specific dimensions: encoder input (D=32, N=2048), FP0 (D=256), FP1 (D=256), FP2 (D=128), FP3 (D=64).

---

## 8. Loss Computation Block *(Training Only)*

The **Loss Computation block** takes the predicted velocity **v̂ [B, 3, 2048]** from the backbone, the target velocity **(x1 − x0) [B, 3, 2048]**, and the conditioning tokens **Z [B, 64, 512]** plus the AE encoding of the clean x1 as inputs. It computes two terms.

**Term 1 — CFM Velocity Loss:** The mean squared error between the predicted velocity v̂ and the ground-truth velocity target (x1 − x0). This is the core OT-CFM objective: the backbone is trained to predict the straight-line displacement vector that maps the interpolated state x_t to the clean endpoint x1.

**Term 2 — Latent Consistency Loss:** Encourages the conditioning tokens to be aligned with the clean geometry even when computed from a noisy input. It computes the AE encoding of the clean patch x1 to get clean tokens [B, 64, 512], mean-pools both the predicted tokens Z and the clean tokens to **[B, 512]** vectors, normalises them to the unit sphere, and computes the cosine dissimilarity **1 − cos(Z̄, Z̄_clean)**. This loss ensures that the FreqEncodingTransformer's output encodes a semantically plausible latent representation of the target shape, not just an arbitrary transformation of the noisy encoding.

The total training loss is:

```
L = L_CFM + 0.3 × L_consistency
```

The weight 0.3 was chosen to make the consistency loss a secondary regulariser rather than the primary objective.

> **draw.io note:** Draw this as a rectangle labelled `"Loss = CFM Loss + 0.3 × Latent Consistency Loss"`. Show three input arrows: `"v̂ [B,3,2048]"` from the backbone, `"(x1−x0) target [B,3,2048]"` from the interpolation block, and `"Z̄ vs AE(x1)̄"` for the consistency term. One output arrow labelled `"scalar L"` exits the block.

---

## 9. Euler ODE Solver Block *(Inference Only)*

During inference the **Euler ODE Solver** replaces the training pipeline. It receives the original noisy input **x_start [B, 3, 2048]** and runs **10 integration steps** with step size Δt = 0.1. Before the loop begins, the frozen AE encodes x_start once to produce **ae_latent [B, 64, 512]**, which is reused at every step (the AE encoding does not change during inference because x_start is fixed).

In each step i (for i = 0, 1, ..., 9):

1. The current time **t = i/10** is broadcast to shape [B] and scaled to t·1000
2. The FreqEncodingTransformer transforms ae_latent with this scaled timestep to produce step-specific conditioning tokens **Z_i [B, 64, 512]**
3. The backbone predicts the velocity **v_i = PVCNN2Unet(x_t, t·1000, Z_i) [B, 3, 2048]**
4. The state is updated as **x_{t+Δt} = x_t + v_i · Δt**

After 10 steps, x_t is the denoised output.

> **draw.io note:** Draw this as a loop block labelled `"Euler ODE (10 steps, Δt=0.1)"`. Inside the loop body, show the sequential calls: `"AE encode (once)"` → `"FreqTransformer(ae_latent, t_i)"` → `"PVCNN2Unet(x_t, t_i, Z_i)"` → `"x_t ← x_t + v_i·Δt"`. Use a loop-back arrow to indicate the 10 iterations. The block receives `"x_start [B,3,2048]"` and emits `"x_denoised [B,3,2048]"`.

---

## 10. Full Data Flow Summary

The complete data flow through the system during **training** is as follows:

1. A batch of noisy and clean patches enters the **OT Coupling** block, which reorders them to minimise transport cost.
2. A random timestep t is sampled and the **Interpolation** block computes x_t.
3. The noisy input x0 is passed to the **frozen SemanticAE**, which encodes it to AE latent tokens [B, 64, 512].
4. These tokens and the scaled timestep t·1000 are passed to the **FreqEncodingTransformer**, which outputs conditioning tokens Z [B, 64, 512].
5. The interpolated state x_t, the timestep, and the conditioning tokens Z are passed to the **PVCNN2Unet**.
6. At the encoder input, Z injects into the feature map via **write_attn** before the SA layers process it.
7. As the SA layers downsample the point cloud, a global mean-pool of Z provides **FiLM conditioning** at every block.
8. At the bottleneck, **LinearAttention** captures global spatial context.
9. As the FP decoder layers upsample back to 2048 points, Z injects again at each of the four FP levels via separate **write_attn** instances.
10. The decoder output [B, 32, 2048] passes through the output MLP to produce the velocity field **v̂ [B, 3, 2048]**.
11. The **Loss block** computes the CFM loss against the ground-truth velocity and adds 0.3 times the cosine consistency loss.
12. Gradients flow through the backbone, write_attn modules, and FreqEncodingTransformer, but **not** through the SemanticAE.

During **inference**, OT coupling and interpolation are skipped. The frozen AE encodes the noisy input once. The Euler loop runs 10 times, calling FreqTransformer and backbone at each step with the current t, progressively denoising x_t toward the clean shape.

---

## 11. Module Dimensions at a Glance

The following table lists every module block with its input tensor shape, output tensor shape, and parameter count. Use these as arrow labels and block annotations in draw.io.

| Block | Input Shape | Output Shape | Parameters | Trainable |
|---|---|---|---|---|
| OT Coupling | [B,3,2048], [B,3,2048] | [B,3,2048], [B,3,2048] | 0 | — |
| Interpolation | [B,3,2048], [B,3,2048], t | [B,3,2048] | 0 | — |
| SemanticAutoencoder | [B,3,2048] | [B,64,512] | 5,985,653 | No |
| FreqEncodingTransformer | [B,64,512], [B] | [B,64,512] | 17,389,184 | Yes |
| LatentWriteAttention (encoder) | [B,32,2048], [B,64,512] | [B,32,2048] | ~131K | Yes |
| LatentWriteAttention (FP ×4) | [B,D,N], [B,64,512] | [B,D,N] | ~235K each | Yes |
| PVCNN2Unet (SA + FP + MLP) | [B,3,2048], t, Z | [B,3,2048] | ~19.4M | Yes |
| Loss (CFM + consistency) | v̂, target, Z | scalar | 0 | — |
| Euler ODE Solver | [B,3,2048] | [B,3,2048] | 0 (uses above) | — |
```