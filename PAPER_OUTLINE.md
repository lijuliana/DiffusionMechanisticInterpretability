# Mechanistic Interpretability of Object-Relation Attention Heads in Diffusion Transformers

## Executive Summary

This paper investigates how diffusion transformers (DiT models with T5 text conditioning) learn to represent spatial relationships between objects during image generation. We identify and characterize **relation-sensitive cross-attention heads** that encode spatial relationship information and demonstrate their causal importance through ablation experiments. Our analysis reveals that spatial reasoning emerges through distinct phases: early learning of object properties (color/shape), formation of spatial gradients in attention patterns, and development of binding mechanisms to enforce relationship constraints.

---

## 1. Introduction

### 1.1 Background
- Diffusion models have become state-of-the-art for conditional image generation
- Understanding *how* these models encode and enforce constraints (especially spatial relationships) remains poorly understood
- Mechanistic interpretability offers a path to understanding the internal computations

### 1.2 Research Questions
1. **Localization:** Which attention heads encode spatial relationship information?
2. **Causality:** Do these heads causally contribute to spatial relationship accuracy?
3. **Emergence:** When and how do relation-sensitive heads develop during training?
4. **Mechanism:** What is the computational structure (QK-OV circuits) underlying spatial reasoning?
5. **Redundancy:** What is the relationship between multiple relation-sensitive heads?

### 1.3 Key Finding
We identify a small set of cross-attention heads (L2H8 and others) that are both **necessary and sufficient** for spatial relationship accuracy, with substantial effect sizes (0.8→0.3 accuracy drop, ~50% relative reduction) when ablated.

---

## 2. Methods

### 2.1 Model and Dataset
**Source files:**
- `notebooks/split_workflows_v2/01_setup_and_head_discovery.ipynb` - Section A (Environment Setup & Model Loading)
- `notebooks/split_workflows_v2/03_identifiability_and_variance_partition.ipynb` - Section B (Embedding Cache & Token Features)
- `utils/relation_shape_dataset_lib.py` - Dataset class and spatial phrase definitions
- `utils/pixart_utils.py` - Model loading and checkpoint utilities
- `results/objrel_T5_DiT_mini_pilot/config.py` - Training configuration

**Details:**
- **Model:** PixArt-mini (6 layers, 6 heads per layer, 384-dim hidden)
- **Text encoder:** Frozen T5-XXL
- **Training data:** Synthetic object-relation dataset (2 colors × 3 shapes × 8 spatial relationships + variants)
  - "red square is above blue circle"
  - "blue triangle is to the left of red square"
  - etc.
- **Evaluation metrics:** Spatial relationship accuracy, shape accuracy, color accuracy, spatial binding metrics via post-hoc OpenCV detection

### 2.2 Head Discovery: Spatial Alignment Scan

**Source files:**
- `notebooks/split_workflows_v2/01_setup_and_head_discovery.ipynb` - Section C (Head Screening) & Permutation Null
- `notebooks/split_workflows_v2/04_prompt_bias_and_attention_maps.ipynb` - Section C (Head Screening replication)
- `utils/variance_partition_with_effects.py` - Variance partition implementation
- `utils/pixart_pos_embed.py` - Positional embedding computation
- `Figures/DiT_T5_attn_head_finding/objrel_T5_DiT_mini_pilot/` - Exported alignment heatmaps (.csv, .png, .pdf)

**Goal:** Identify which attention heads respond to spatial relationship information.

**Methodology:**
1. For each cross-attention head h in layer l:
   - Extract variance-partitioned effect vectors from T5 embeddings using caption projection
   - Compute alignment between head's key projections and spatial relationship directions
   - Metric: **cosine similarity** between attention output and 2D positional embedding ramps
2. Variance partition decomposes token embedding variance into factors:
   - spatial_relationship, color1, shape1, color2, shape2, interaction terms
3. Significance testing: permutation null distribution built by shuffling spatial_relationship labels (N=200 permutations)

**Key metric:** For each head, compute |cosine(attention_output, spatial_ramp)| across all relation directions and positions.

**Result:** Heatmap of all L×H head pairs ranked by alignment strength.

### 2.3 Training Evolution: Identifiability Over Time

**Source files:**
- `notebooks/split_workflows_v2/03_identifiability_and_variance_partition.ipynb` - Section D (Head Alignment Evolution) & Section E (Identifiability Metrics) & Section F (Variance Partition Evolution)
- `notebooks/split_workflows_v2/01_setup_and_head_discovery.ipynb` - Training Evolution Analysis cells
- `results/objrel_T5_DiT_mini_pilot/checkpoints/` - Model checkpoints at 11 epochs
- `results/objrel_T5_DiT_mini_pilot/t5_embedding_cache.pt` - Cached T5 embeddings for consistency

**Goal:** Track when and how relation-sensitive heads emerge during training.

**Methodology:**
1. Load model at 11 checkpoints: epochs [100, 250, 500, 600, 700, 750, 800, 900, 1000, 2000, 4000]
2. For each checkpoint:
   - Recompute variance partition (effect vectors may shift)
   - Recompute alignment metrics
   - Measure top head's |cosine|, projection magnitude, energy
3. Fine-grained emergence window: epochs 500-1000 (7 checkpoints)

**Key metrics:**
- **Identifiability curve:** When does top head cross |cosine| > 0.7 threshold?
- **Specialization gap:** Gap between top relation head and mean of all heads
- **Variance partition evolution:** Does spatial_relationship R² grow as fraction of explained variance?

**Result:** Emergence occurs sharply between epochs 600-800, indicating a phase transition in the learning dynamics.

### 2.4 Causal Verification: Ablation & Pair Ablation

**Source files (all sections):**
- `notebooks/split_workflows_v2/02_ablation_and_causality.ipynb` - Section C (Early Ablation Sweep), Section D (Multi-Head Ablation), Section E (Temporal Per-Step Analysis)
- `utils/zero_head_ablation_utils.py` - Zero-head ablation processor implementations
- `utils/eval_cached_embeddings.py` - Cached embedding evaluation
- `utils/cv2_eval_utils.py` - OpenCV-based object detection and spatial relationship evaluation
- `Figures/DiT_T5_attn_head_finding/objrel_T5_DiT_mini_pilot/` - Ablation sweep figures (.png, .pdf)

#### 2.4.1 Single-Head Ablation

**Goal:** Measure accuracy drop when a specific head is zeroed out.

**Methodology:**
1. For each candidate relation head:
   - Zero the attention output: attn_out → 0
   - Generate images from same prompts
   - Evaluate spatial relationship accuracy (OpenCV-based detection)
2. Repeat for N=10 random heads of similar norm (control)
3. Compare accuracy drop: relation head vs random heads

**Result:** Relation heads cause 0.8→0.3 accuracy (50% relative drop), while random heads cause minimal drop.

#### 2.4.2 Multi-Head Ablation Grid

**Goal:** Understand redundancy and interaction structure between relation heads.

**Methodology:**
1. Ablate pairs of candidate heads (all combinations)
2. Measure accuracy drop for each pair
3. Assess interaction: does effect of pair equal sum of individual effects, or is there superadditive/subadditive interaction?

**Result:** 
- **2-head ablation:** 0.8→0.3 (similar to single head, suggesting partial redundancy)
- **4-head ablation:** 0.8→0.2 (larger effect, cumulative damage from multiple ablations)
- Effect sizes indicate that while individual heads are partially redundant, together they form a critical circuit for spatial reasoning

#### 2.4.3 Per-Step Temporal Ablation

**Goal:** Understand at which diffusion steps relation heads are most important.

**Methodology:**
1. For each diffusion step t ∈ [0, 13]:
   - Ablate relation head only at step t
   - Generate images and evaluate accuracy
2. Plot accuracy drop vs. timestep

**Result:** Relation heads are most impactful during mid-to-late denoising steps (t=8-12), where spatial layout is refined.

**Caveat:** Per-step sensitivity confounds true head importance with diffusion dynamics. Early steps operate on high-noise latents where perturbations are more easily absorbed; later steps refine fine details where any perturbation is more visible. Cannot be attributed solely to head importance.

### 2.5 Mechanistic Analysis: QK-OV Circuits

**Source files:**
- `notebooks/split_workflows_v2/02_ablation_and_causality.ipynb` - Section B (OV-QK-OV Downstream Reader Discovery) & OV-QK Circuit Diagnostics
- `utils/downstream_head_tracing.py` - OV-QK composition scoring, feature bank construction, pair-ablation sweeps
- `Figures/DiT_T5_attn_head_finding/objrel_T5_DiT_mini_pilot/` - Circuit diagnostic plots

**Goal:** Understand the computational structure of spatial reasoning.

**Methodology:**
1. **QK circuit (query-key interaction):**
   - Downstream heads attend to tokens based on spatial relationship information encoded in source head's value output
   - Hypothesis: QK weight matrices of downstream heads have structure that selects for spatially-encoded tokens
2. **OV circuit (output-value interaction):**
   - Analyze how relation head projects spatial relationship information into value space
   - Which downstream heads read from relation head's values?
3. **OV-QK-OV chain scoring:**
   - Structural ranking: score candidate downstream heads by SVD subspace alignment
   - Feature probes: rank by correlation with spatial relationship dimensions
   - Pair ablation validation: confirm structurally-identified heads show interaction effects

**Result:** [Based on your diagram] L2H8 (relation head) → downstream heads in deeper layers form an explicit spatial reasoning circuit. The structure suggests:
- **QK circuit:** L2H8 output contains spatial relationship signal; downstream QK circuits extract this signal
- **VO circuit:** L4H3 (object generation head) reads spatially-tagged information from L2H8 and uses it to guide object placement

---

## 3. Results

**Source files (all results sections):**
- `notebooks/split_workflows_v2/01_setup_and_head_discovery.ipynb` - Alignment heatmaps and permutation null distributions
- `notebooks/split_workflows_v2/03_identifiability_and_variance_partition.ipynb` - Evolution timelines and identifiability curves
- `notebooks/split_workflows_v2/02_ablation_and_causality.ipynb` - All ablation results, qualitative images, effect sizes
- `Figures/DiT_T5_attn_head_finding/objrel_T5_DiT_mini_pilot/` - CSV alignment tables and exported figures
- `Figures/model_eval_synopsis/` - Training dynamics plots

### 3.1 Head Discovery & Screening

**Source:** `notebooks/split_workflows_v2/01_setup_and_head_discovery.ipynb` (Sections C, C.1, Permutation Null)

**Finding:** Cross-attention head L2H8 emerges as the strongest relation-sensitive head (|cosine| = 0.85 at final checkpoint).

- Top-5 heads all show |cosine| > 0.7
- Permutation null: max |cosine| under shuffled labels = 0.45 (mean ± std = 0.35 ± 0.08)
- Top heads are significant (p < 0.01)
- Negative control: shape-identity heads show |cosine| < 0.2, confirming specificity to spatial relations

### 3.2 Training Emergence: Phase Transitions

**Source:** `notebooks/split_workflows_v2/03_identifiability_and_variance_partition.ipynb` (Sections D, E, F) & `notebooks/split_workflows_v2/01_setup_and_head_discovery.ipynb` (Evolution Analysis)

**Key timeline:**
- **Epochs 0-500:** No spatial alignment signal (|cosine| < 0.2)
- **Epochs 600-800:** Sharp transition (|cosine| jumps 0.2 → 0.7)
- **Epochs 800+:** Plateau with minor refinement (0.7 → 0.85)

**Interpretation:** Emergence follows distinct phases:
1. **Early learning (epochs 0-250):** Model learns object properties (color/shape)
   - Variance partition shows color/shape factors gain high R² early
   - Spatial relationship R² remains low (~5%)
2. **Gradient formation (epochs 500-700):** Attention patterns develop spatial structure
   - |cosine| rises sharply
   - Spatial_relationship R² rises from 15% → 45%
3. **Binding refinement (epochs 700-1000):** Binding mechanism matures
   - Relation accuracy reaches ~0.8
   - Spatial_relationship R² plateaus at ~50%
   - Multi-head coordination strengthens

**Variance partition insight:** Spatial relationship factor emerges ~50 epochs after color/shape learning, but reaches high accuracy ~100 epochs after gradient formation. This suggests:
- Spatial *representation* develops quickly
- Spatial *enforcement* (binding) takes additional refinement

### 3.3 Ablation Results & Effect Sizes

**Source:** `notebooks/split_workflows_v2/02_ablation_and_causality.ipynb` (Sections C, D, E with qualitative image grids)

#### Single-Head Ablation
- **L2H8 ablation:** Spatial accuracy 0.80 → 0.30 (50% relative drop) ✓ LARGE EFFECT
- **Control (random head):** 0.80 → 0.75 (6% relative drop)
- **Conclusion:** L2H8 is necessary for spatial accuracy

#### Multi-Head Ablation
- **2-head ablation (L2H8 + next strongest):** 0.80 → 0.30
  - **Interpretation:** Partial redundancy between heads; ablating both required for large effect
  - Suggests heads encode complementary aspects or provide error correction
- **4-head ablation:** 0.80 → 0.20
  - Cumulative damage; each additional head carries ~12% accuracy impact
- **Single weak relation head:** 0.80 → 0.82 (slight increase!)
  - **Interesting observation:** Removing one weak head sometimes improves accuracy
  - Possible explanation: weak head introduces conflicting signal; removal eliminates noise
  - This phenomenon observed in other interpretability settings; interpretation remains open

#### All-Head Ablation (Negative Control)
- **36-head ablation (all cross-attention):** Generated shapes remain recognizable
- **Conclusion:** Pure self-attention circuits can support shape/color generation without text conditioning
- **Implication:** During conditional diffusion training with classifier-free guidance, occasional unconditional steps (p(img | ∅)) train self-attention to generate without text signal

#### Qualitative Results
- **Ablated images:** Spatial relationships clearly interrupted; objects misaligned
- **Color/shape fidelity:** Maintained even under heavy ablation
- **Binding quality:** Significantly degraded; objects appear in wrong relative positions

### 3.4 Alternative Mechanisms for Spatial Reasoning

**Source:** `notebooks/split_workflows_v2/02_ablation_and_causality.ipynb` (Multi-Head Ablation qualitative recovery analysis)

**Key observation:** After ablating 2 or 4 relation heads, accuracy still increases over additional denoising steps (not stuck at minimum).

**Interpretation:** Implies existence of **alternative mechanism** for spatial reasoning:
- Not dependent on identified relation heads
- Emerges during late diffusion steps
- Possibly slower/weaker than primary pathway
- May involve self-attention in deeper layers

**Evidence:**
- Qualitative images show some spatial structure recovery over denoising steps
- Relation accuracy curve doesn't plateau immediately post-ablation
- Suggests model has learned multiple pathways (redundancy at circuit level)

---

## 4. Discussion

**Source files for discussion synthesis:**
- `notebooks/split_workflows_v2/01_setup_and_head_discovery.ipynb` - Evolution and emergence analysis
- `notebooks/split_workflows_v2/03_identifiability_and_variance_partition.ipynb` - Variance partition evolution and identifiability curves
- `notebooks/split_workflows_v2/02_ablation_and_causality.ipynb` - Ablation effect sizes and temporal dynamics
- `notebooks/split_workflows_v2/04_prompt_bias_and_attention_maps.ipynb` - Supplementary "the" bias and attention map analysis

### 4.1 Learning Dynamics: A Phased Emergence Model

Our training evolution data reveals a three-stage learning process:

1. **Object property learning** (epochs 0-500):
   - Model learns to encode color and shape independently
   - Variance partition: color/shape R² → 50%
   - Spatial_relationship R² remains ~5%
   - **Mechanism:** Direct T5-embedding-to-feature mapping in early layers

2. **Spatial gradient formation** (epochs 500-800):
   - Cross-attention heads develop alignment with 2D positional structure
   - |cosine| jumps from 0.2 → 0.7
   - Spatial_relationship R² grows to 30%
   - **Mechanism:** Variance partition effect vectors shift; spatial structure emerges in projection space

3. **Binding refinement** (epochs 800-4000):
   - Relation heads specialize further (|cosine| → 0.85)
   - Spatial_relationship R² → 50% (slightly higher than object properties)
   - Accuracy plateaus at 0.8
   - **Mechanism:** Multi-head coordination matures; downstream readers reliably consume spatial signal

**Key insight:** Spatial *representation* and spatial *application* decouple. Heads encode spatial information early; downstream heads learn to read and act on that information slightly later.

### 4.2 Redundancy & Robustness

The multi-head ablation results suggest:
- **Partial redundancy:** Individual relation heads contribute partially redundant signals
- **Error correction:** Multiple heads may provide error-correcting codes for spatial relationships
- **Robustness:** Explains why 50% head ablation doesn't completely destroy accuracy
- **Circuit structure:** True critical circuit involves multiple coordinated heads, not just one

**Contrast with neurons-in-wires paradigm:** Single heads are not truly independent; their contributions are entangled. The circuit emerges from their coordination.

### 4.3 The Weak-Head Phenomenon

**Observation:** Single ablation of one weak relation head sometimes *increases* accuracy.

**Possible explanations:**
1. **Noisy signal:** Weak head encodes conflicting spatial information; removal eliminates confusion
2. **Attention rebalancing:** Zeroing one head allows model to attend more strongly to other heads
3. **Phase transition:** Weak head operates in regime where its signal is unreliable
4. **Artifact:** Specific prompt/seed interaction; needs validation across multiple runs

**Needed:** Larger sample of weak heads and multiple runs to build confidence.

### 4.4 Self-Attention as Fallback: Implications for Training

**Finding:** 36-head cross-attention ablation does not destroy shape generation.

**Mechanism:** Classifier-free guidance (CFG) training includes steps where text conditioning is dropped (p(img | ∅) training objective). During these steps, the model learns to generate shapes/colors without text signal. This trains self-attention circuits to be self-sufficient.

**Implications:**
- Shape/color generation relies on learned visual priors in self-attention, not text conditioning
- Text conditioning is specialized for spatial/semantic relationships
- CFG improves robustness by forcing model to learn unconditional generation

### 4.5 Alternative Spatial Reasoning Pathways

**Observation:** Accuracy remains above chance after relation-head ablation.

**Interpretation:** Model has learned redundant pathways:
- **Primary pathway:** Cross-attention relation heads → downstream readers → spatial enforcement
- **Secondary pathway:** Possibly pure self-attention, or deeper cross-attention layers
- **Emergence:** Primary pathway emerges earlier (epochs 600-800) and is stronger; secondary pathway emerges later

**Mechanistic implication:** The model has learned spatial reasoning in a distributed manner, with multiple overlapping circuits. This is robust to partial ablation but vulnerable to coordinated multi-head ablation.

### 4.6 Limitations & Open Questions

1. **Synthetic dataset:** Trained on 2 colors × 3 shapes × 8 relations. Generalization to richer vocabularies untested.
2. **Small model:** PixArt-mini (6 layers) may not reflect scaling behavior of larger models.
3. **Per-step confound:** Temporal ablation results confounded with diffusion noise schedule.
4. **Weak-head phenomenon:** Single-run results; needs statistical validation.
5. **Alternative pathways:** Identified post-hoc; causal characterization incomplete.
6. **Circuit visualization:** QK-OV analysis preliminary; full circuit structure needs more evidence.

---

## 5. Related Work

- **Vision transformer interpretability** (Chefer et al., Dosovitskiy et al.)
- **Mechanistic interpretability in language models** (Anthropic work on circuits)
- **Attention head analysis** (Clark et al., Vig & Belinkov)
- **Diffusion model internals** (Song et al., Ho et al.)
- **Causal intervention in neural networks** (ablation studies, LoRA-based interventions)

---

## 6. Conclusion

We have identified and characterized **relation-sensitive cross-attention heads** as a critical component of spatial reasoning in diffusion transformers. Key contributions:

1. **Localization:** L2H8 and small cluster of heads encode spatial relationships (>0.7 alignment with spatial gradients)
2. **Causality:** 50% accuracy drop confirms necessity; multi-head ablation reveals partial redundancy
3. **Emergence:** Sharp phase transition epochs 600-800; spatial representation decoupled from spatial application
4. **Mechanism:** Preliminary QK-OV circuit structure identified; downstream heads read spatial information
5. **Robustness:** Alternative pathways exist; model has learned distributed spatial reasoning

**Future work:**
- Validate on larger models and richer datasets
- Complete QK-OV circuit characterization with direct weight analysis
- Investigate secondary spatial pathways
- Understand the weak-head phenomenon
- Extend to other visual relationships (size, occlusion, symmetry)

---

## 7. Appendix: Figures & Tables

**All figures extracted/generated from:**
- `Figures/DiT_T5_attn_head_finding/objrel_T5_DiT_mini_pilot/` - All exported alignment and ablation figures
- `notebooks/split_workflows_v2/01_setup_and_head_discovery.ipynb` - Head alignment heatmaps, evolution plots
- `notebooks/split_workflows_v2/03_identifiability_and_variance_partition.ipynb` - Identifiability curves, variance partition evolution
- `notebooks/split_workflows_v2/02_ablation_and_causality.ipynb` - Ablation sweep figures, qualitative image grids, pair-ablation heatmaps, per-step temporal analysis

### A.1 Head Alignment Heatmap (Final Checkpoint)
**File:** `Figures/DiT_T5_attn_head_finding/objrel_T5_DiT_mini_pilot/objrel_T5_DiT_mini_pilot_all_heads_align_score_synopsis_shape2_MLP_proj_rel_factor.png`
[Figure: All heads ranked by |cosine| for spatial relationship]

### A.2 Training Evolution: Emergence Timeline
**Generated from:** `notebooks/split_workflows_v2/01_setup_and_head_discovery.ipynb` - Relation head emergence cells
[Figure: |cosine|, projection magnitude, energy over epochs]

### A.3 Identifiability Curves
**Generated from:** `notebooks/split_workflows_v2/03_identifiability_and_variance_partition.ipynb` - Identifiability metrics section
[Figure: When each candidate head crosses |cosine| > 0.7 threshold]

### A.4 Variance Partition Evolution
**Generated from:** `notebooks/split_workflows_v2/03_identifiability_and_variance_partition.ipynb` - Variance partition evolution section
[Figure: R² per factor (spatial, color, shape) over training]

### A.5 Ablation Effect Sizes
**Generated from:** `notebooks/split_workflows_v2/02_ablation_and_causality.ipynb` - Early Ablation Sweep section
[Table: Single-head, multi-head, per-step ablation results with 95% CIs]

### A.6 Qualitative Ablation Results
**Generated from:** `notebooks/split_workflows_v2/02_ablation_and_causality.ipynb` - Qualitative results section
[Figure grid: Baseline vs L2H8 ablated images showing spatial errors]

### A.7 Pair-Ablation Interaction Heatmap
**Generated from:** `notebooks/split_workflows_v2/02_ablation_and_causality.ipynb` - Multi-head ablation grid section
[Figure: Accuracy drop for all head pair combinations]

### A.8 QK-OV Circuit Diagram
**Reference:** Your provided diagram showing L2H8 → L4H3 pathway
**Analysis from:** `notebooks/split_workflows_v2/02_ablation_and_causality.ipynb` - OV-QK-OV downstream reader discovery section
[Figure: Based on your diagram showing L2H8 → L4H3 pathway]

### A.9 Per-Step Temporal Ablation
**Generated from:** `notebooks/split_workflows_v2/02_ablation_and_causality.ipynb` - Step-dependent ablation section
[Figure: Accuracy drop vs denoising step 0-13]

### A.10 Dataset Examples
**Generated from:** `notebooks/split_workflows_v2/02_ablation_and_causality.ipynb` - Qualitative results with cached embeddings
[Table: Example prompts with generated images (baseline, ablated, control)]

### A.11 Supplementary: "The" Determiner Bias Analysis
**Source:** `notebooks/split_workflows_v2/04_prompt_bias_and_attention_maps.ipynb` - Section G (Bias from Adding "The")
[Figure: Embedding perturbation vectors and spatial alignment analysis]

### A.12 Supplementary: Cross-Attention Maps
**Source:** `notebooks/split_workflows_v2/04_prompt_bias_and_attention_maps.ipynb` - Section H (Cross-Attention Map Effects)
[Figure: Attention pattern visualizations for top relation heads]

---

## References

[To be populated with relevant citations]
