# NeurIPS Frontier Improvements: High-Level Prompts for Claude Code

## Goal
Advance the current 16-page paper on mechanistic interpretability of diffusion transformers from solid analysis to frontier contribution by implementing four major additions: circuit analysis, intervention experiments, visualization, and robust evaluation.

---

## Improvement 1: Deep Circuit Mechanistic Analysis
**What to do**: Map out how spatial information flows through the network at a detailed mechanistic level.

**Create**: New notebook `05_circuit_tracing.ipynb`

**Include**:
- Path tracing from input through layers to output (activation magnitudes, effective rank, information preservation)
- SVD analysis of QK and OV matrices for top spatial heads vs. control heads (what's the difference in how information is structured?)
- Feature attribution: trace spatial factors back to T5 embedding space (which T5 dimensions encode "above", "left", etc?)
- Synthesize findings into a plain-language description of what the circuit actually computes

**Output**: 4-5 figures, 2-3 summary tables, and a circuit description document explaining the computation step-by-step

**Why it matters**: Shows not just that heads are necessary (causality) but HOW they work mechanistically. This moves from correlation-based analysis to true mechanistic understanding.

---

## Improvement 2: Intervention Experiments (Steering)
**What to do**: Demonstrate that the spatial circuit is not just correlated with spatial reasoning but sufficient and predictable. Show that you can steer spatial relationships by manipulating attention head outputs.

**Create**: New notebook `05b_intervention_steering.ipynb`

**Include**:
- Amplification experiments: systematically amplify spatial head outputs and measure effects on accuracy and image quality
- Signal swapping: take spatial information from one prompt, inject into another, verify composition works
- Selective steering: amplify one spatial relationship (e.g., "left") while suppressing its opposite, show objects move in predictable directions
- Per-step importance: understand which diffusion steps are most sensitive to spatial head manipulation

**Output**: 4-5 figures showing intervention effects, sample images, accuracy curves

**Why it matters**: Proves the circuit is not just necessary but sufficient and predictable. This is the difference between "we found a correlation" and "we understand and can control the mechanism."

---

## Improvement 3: Visualization & Temporal Analysis
**What to do**: Create publication-quality visualizations and analyze when different components of the circuit emerge during training.

**Create**: New notebook `05c_visualization_temporal.ipynb`

**Include**:
- Attention heatmaps showing which text tokens attend to which spatial image regions
- Saliency maps: pixel-wise analysis of which image regions are affected by spatial heads
- Checkpoint evolution: track when downstream readers learn to read from relation heads across training
- Beautiful, interpretable figures suitable for a top-tier venue

**Output**: 3-4 figures, temporal evolution plots, emergence timeline

**Why it matters**: Makes the mechanism visually clear and provides evidence of the learning dynamics.

---

## Improvement 4: Robust Evaluation & Statistical Rigor
**What to do**: Ensure all claims are backed by rigorous statistics, not just single runs or post-hoc analysis.

**Create**: New notebook `05d_robust_evaluation.ipynb`

**Include**:
- Multi-seed ablation experiments (run ablations with different random seeds, compute confidence intervals)
- Statistical significance testing with appropriate corrections for multiple comparisons
- Effect size quantification (Cohen's d, η²)
- Robustness checks: verify findings hold across different prompt templates, evaluation metrics, ablation methods

**Output**: Tables with confidence intervals, significance test results, effect size summaries

**Why it matters**: NeurIPS-level papers are rigorous about statistics. Single runs aren't enough. This moves from "we see a trend" to "this effect is statistically significant and robust."

---

## Additional Considerations

### Visualization Standards
All figures should be publication-quality:
- Use serif fonts (standard academic style, not AI-generated looking)
- High resolution (300 dpi for PDF export)
- Clear, readable color schemes
- Professional formatting

### Experimental Choices
- Analyze top-3 spatial heads AND top-3 random non-spatial heads (provides both positive results and controls)
- Significance testing is good if not compute-heavy (e.g., Bonferroni-corrected t-tests with ~5 seeds are fine)
- Start with final checkpoint (epoch 4000), then optionally add checkpoint variants

### Integration with Paper
After each notebook is complete:
- Extract figures and tables
- Add corresponding Methods and Results sections to the paper
- Update Discussion to synthesize findings
- Paper should grow from 16 pages → 24-26 pages with all improvements

---

## Optional (Phase 5): Generalization Beyond Synthetic Data
If compute/time permits, validate that findings generalize:
- Run on real image-text pairs (COCO or similar)
- Test on larger models (full PixArt or DiT-L)
- Verify relation heads exist at scale

This would strengthen the paper significantly but is secondary to Improvements 1-4.

---

## Success Criteria

The paper should demonstrate:
1. **Mechanistic understanding**: We can describe what each head computes and why
2. **Predictive control**: We can manipulate spatial relationships intentionally
3. **Visual clarity**: Beautiful figures that make the mechanism obvious
4. **Statistical rigor**: All claims have confidence intervals and significance tests
5. **Generalization**: Results are robust to variations in experimental design

Final paper characteristics:
- 24-26 pages (dense technical content)
- Complete circuit mapping from input to output
- Proof that spatial circuit is sufficient (not just necessary)
- Frontier-level mechanistic insight and rigor
