# DiffusionInterp

This repository contains notebook-first analysis code for studying how a small PixArt/DiT model with T5 text embeddings learns object-relation structure over training. The current workflow is organized around four split notebooks plus a few supporting analysis notebooks, helper utilities, local evaluation scripts, and saved result artifacts.

The main themes are:

- finding relation-sensitive cross-attention heads
- tracing when those heads emerge during training
- testing causality with zero-head ablations
- decomposing text-feature variance into relation / object / prompt factors
- probing prompt bias introduced by function words like `the`
- evaluating checkpoint generalization with cached text embeddings

## Notebook Inventory

### Split workflow notebooks

#### `notebooks/split_workflows/01_setup_and_head_discovery.ipynb`
Purpose:
- environment setup and model loading
- build/load cached embeddings and token-level features
- head screening via alignment scans
- selective head visualization
- relation-head emergence over training

Key sections:
- `Section A - Environment Setup and Model Loading`
- `Section B - Build/Load Embedding Cache and Token Features`
- `Section C - Head Screening (Alignment Scan)`
- `Trace spatial heads through training`
- `Relation head emergence (epochs 500–1000)`

Typical outputs:
- all-head alignment CSV-style tables
- head heatmaps for shape/object alignment
- training-trajectory plots for candidate relation heads

#### `notebooks/split_workflows/02_ablation_and_causality.ipynb`
Purpose:
- causal testing of discovered heads
- early OV-QK-OV downstream-reader search
- ranking comparison for alternate downstream-head hypotheses
- ablation sweeps and head-importance summaries

Key sections:
- `Section A - Environment Setup and Model Loading`
- `Section B - Build/Load Embedding Cache and Token Features`
- `Downstream readers (early): OV-QK-OV candidate discovery`
- `OV-QK circuit diagnostics`
- `Ranking comparison: write-aware alternatives`
- `Ablation experiments`
- `Relation head ablation results`
- `Comparison: relation head vs random head ablation`

Typical outputs:
- candidate downstream-head ranking tables
- OV-QK read/write diagnostic plots
- write-aware ranking comparison tables and heatmaps
- early ablation sweep figures
- pair/multi-head ablation summaries

Notes:
- this notebook now includes an intentional stop before later sections so `Run All` can halt after the OV-QK analysis if desired

#### `notebooks/split_workflows/03_identifiability_and_variance_partition.ipynb`
Purpose:
- analyze when the relation representation becomes identifiable
- track projection norms and attention changes over training
- run variance partition experiments on token embeddings and projected features

Key sections:
- `Identifiability and variance partition (evolution analysis)`
- `Alternate candidate discovery: Method A (OV-QK composition)`
- `When does the relation head become identifiable?`
- `Projection norms and attention patterns over training`
- `Variance partition experiments`

Typical outputs:
- time-course plots of head identifiability
- projection norm trajectories
- variance partition tables and plots
- `effect_vecs` / `levels_map` objects used downstream

#### `notebooks/split_workflows/04_prompt_bias_and_attention_maps.ipynb`
Purpose:
- sanity-check inference
- prompt wording bias analysis
- attention map analysis and visualization

Key sections:
- `Section F - Sanity Check Inference (Model Test)`
- `Section G - Bias from Adding "the" in Prompts`
- `Section H - Cross-Attention Map Effects`

Typical outputs:
- prompt-template comparison tables
- representation-bias similarity analyses
- attention-map visualizations or exported data

### Additional notebooks

#### `notebooks/20260126_T5_word_embed_var_factorization_head_finding_MINI.ipynb`
Monolithic, earlier end-to-end notebook covering setup, head discovery, ablations, qualitative results, perturbation sensitivity, and compensation experiments.

Useful when:
- you want the original all-in-one analysis flow
- you need legacy code paths that the split notebooks refactor

#### `notebooks/checkpoint_training_trajectory_analysis.ipynb`
Training-dynamics notebook focused on checkpoint evolution rather than head-level intervention.

Covers:
- per-layer norm evolution
- head specialization over training
- weight change rates
- EMA vs raw weights
- TensorBoard loss curves
- validation image summaries
- generalization metric trajectories

## Supporting Files

### Analysis utilities in `utils/`

#### `utils/downstream_head_tracing.py`
Notebook-oriented utilities for downstream-head tracing.

Main responsibilities:
- OV-QK and OV-QK-OV candidate scoring
- feature-bank construction from token embeddings
- pair-ablation sweeps
- causal summary tables
- downstream ranking diagnostics

This is the main support module behind the new `02_ablation_and_causality.ipynb` OV-QK workflow.

#### `utils/variance_partition_with_effects.py`
Multivariate variance decomposition for categorical factors.

Used to compute:
- `var_part_df`
- `effect_vecs`
- `levels_map`
- total explained variance

#### `utils/eval_cached_embeddings.py`
Single source of truth for cached-embedding evaluation.

Used by:
- notebooks
- local evaluation scripts
- checkpoint generalization evaluation

Returns:
- per-image evaluation DataFrames
- stacked object-detection DataFrames

#### `utils/zero_head_ablation_utils.py`
Cross-attention zero-head ablation utilities for PixArt/DiT models.

Used for:
- early ablation sweeps
- fixed head-set ablations
- restoring processors after interventions

#### `utils/pixart_utils.py`
Bridge between PixArt training checkpoints and Diffusers `Transformer2DModel` / pipeline objects.

Responsibilities:
- construct diffusers-compatible transformer/pipeline objects
- load EMA checkpoint weights
- provide model-architecture config mappings

#### `utils/pixart_sampling_utils.py`
Notebook compatibility re-export for custom PixArt pipeline helpers.

#### `utils/relation_shape_dataset_lib.py`
Synthetic prompt metadata helper for object-relation experiments.

Provides:
- `ShapesDataset`
- canonical spatial phrase mappings

#### `utils/text_encoder_control_lib.py`
Lookup-table text encoder implementations for random-embedding or pilot-model settings.

Useful for:
- non-T5 experiments
- checkpoint evaluation with alternate text encoders

#### `utils/image_utils.py`
Small image helper utilities, currently including:
- PIL image grid creation

#### `utils/attention_map_store_utils.py`
Attention visualization hook stubs for notebook compatibility.

Important:
- this file is intentionally minimal
- map-producing cells need a real `AttnProcessor2_0_Store` implementation or diffusers-compatible replacement

#### `utils/pixart_pos_embed.py`
Support code for positional embedding handling in PixArt-style models.

#### `utils/cv2_eval_utils.py`
OpenCV-based object detection and relation evaluation.

Used for metrics such as:
- `overall`
- `shape`
- `color`
- `exist_binding`
- `unique_binding`
- `spatial_relationship`

#### `utils/ablation_eval_prompts.py`
Prompt definitions used for ablation/evaluation experiments.

### Experimental scripts

#### `experimental_scripts/generalization_profile_eval_cli.py`
General CLI for evaluating model generalization across prompt templates and checkpoints.

Supports:
- multiple model families
- multiple text encoder types
- checkpoint subsets or full checkpoint sweeps
- prompt template variations

#### `experimental_scripts/run_generalization_eval_local.py`
Workspace-local adaptation focused on `objrel_T5_DiT_mini_pilot`.

Designed for:
- Apple Silicon / MPS
- cached T5 embeddings
- local checkpoint evaluation
- training-curve plot generation

## Generated Results and Artifacts

The repository currently contains two main classes of generated outputs:

- `results/`: run artifacts, checkpoints, evaluation tables, cached embeddings, logs
- `Figures/`: exported figures and figure-source tables for the analysis notebooks

### `results/objrel_T5_DiT_mini_pilot/`

#### Run metadata and caches
- `config.py`
- `t5_embedding_cache.pt`
- `train_log.log`

#### Checkpoints
- `checkpoints/epoch_100_step_4000.pth`
- `checkpoints/epoch_250_step_10000.pth`
- `checkpoints/epoch_500_step_20000.pth`
- `checkpoints/epoch_600_step_24000.pth`
- `checkpoints/epoch_700_step_28000.pth`
- `checkpoints/epoch_750_step_30000.pth`
- `checkpoints/epoch_800_step_32000.pth`
- `checkpoints/epoch_900_step_36000.pth`
- `checkpoints/epoch_1000_step_40000.pth`
- `checkpoints/epoch_2000_step_80000.pth`
- `checkpoints/epoch_4000_step_160000.pth`

These are the model states used throughout the training-trajectory, head-discovery, and evaluation notebooks.

#### Generalization evaluation tables

Per-checkpoint aggregate CSVs:
- `generalization_eval/eval_epoch_100_step_4000.pth.csv`
- `generalization_eval/eval_epoch_250_step_10000.pth.csv`
- `generalization_eval/eval_epoch_500_step_20000.pth.csv`
- `generalization_eval/eval_epoch_600_step_24000.pth.csv`
- `generalization_eval/eval_epoch_700_step_28000.pth.csv`
- `generalization_eval/eval_epoch_750_step_30000.pth.csv`
- `generalization_eval/eval_epoch_800_step_32000.pth.csv`
- `generalization_eval/eval_epoch_900_step_36000.pth.csv`
- `generalization_eval/eval_epoch_1000_step_40000.pth.csv`
- `generalization_eval/eval_epoch_2000_step_80000.pth.csv`
- `generalization_eval/eval_epoch_4000_step_160000.pth.csv`

Consolidated evaluation outputs:
- `generalization_eval/eval_all_checkpoints.csv`
- `generalization_eval/training_curve_metrics.csv`

Object-detection result pickles:
- `generalization_eval/objects_epoch_100_step_4000.pth.pkl`
- `generalization_eval/objects_epoch_250_step_10000.pth.pkl`
- `generalization_eval/objects_epoch_500_step_20000.pth.pkl`
- `generalization_eval/objects_epoch_600_step_24000.pth.pkl`
- `generalization_eval/objects_epoch_700_step_28000.pth.pkl`
- `generalization_eval/objects_epoch_750_step_30000.pth.pkl`
- `generalization_eval/objects_epoch_800_step_32000.pth.pkl`
- `generalization_eval/objects_epoch_900_step_36000.pth.pkl`
- `generalization_eval/objects_epoch_1000_step_40000.pth.pkl`
- `generalization_eval/objects_epoch_2000_step_80000.pth.pkl`
- `generalization_eval/objects_epoch_4000_step_160000.pth.pkl`
- `generalization_eval/objects_all_checkpoints.pkl`

Generated plot files:
- `generalization_eval/plot_all_metrics_over_training.png`
- `generalization_eval/plot_binding_metrics_over_training.png`
- `generalization_eval/plot_spatial_metrics_over_training.png`
- `generalization_eval/plot_temp_exist_and_spatial.png`

These outputs summarize how object existence, binding, and spatial accuracy evolve over training.

#### TensorBoard logs
- `logs/tb_2025-07-17_01_42_04/events.out.tfevents.1752730926.holygpu8a13102.rc.fas.harvard.edu.2826818.2`

### `Figures/DiT_T5_attn_head_finding/objrel_T5_DiT_mini_pilot/`

Head-discovery and ablation figure exports:
- `objrel_T5_DiT_mini_pilot_align_score_allheads_shape1_MLP_proj_rel_factor.csv`
- `objrel_T5_DiT_mini_pilot_align_score_allheads_shape2_MLP_proj_rel_factor.csv`
- `objrel_T5_DiT_mini_pilot_all_heads_align_score_synopsis_shape1_MLP_proj_rel_factor.png`
- `objrel_T5_DiT_mini_pilot_all_heads_align_score_synopsis_shape1_MLP_proj_rel_factor.pdf`
- `objrel_T5_DiT_mini_pilot_all_heads_align_score_synopsis_shape2_MLP_proj_rel_factor.png`
- `objrel_T5_DiT_mini_pilot_all_heads_align_score_synopsis_shape2_MLP_proj_rel_factor.pdf`
- `objrel_T5_DiT_mini_pilot_early_ablation_sweep.png`
- `objrel_T5_DiT_mini_pilot_early_ablation_sweep.pdf`
- `objrel_T5_DiT_mini_pilot_ablation_sweep_best_aligned.png`
- `objrel_T5_DiT_mini_pilot_ablation_sweep_best_aligned.pdf`
- `objrel_T5_DiT_mini_pilot_ablation_sweep_?.png`
- `objrel_T5_DiT_mini_pilot_ablation_sweep_?.pdf`

These figure exports capture:
- all-head alignment synopses
- early ablation sweep behavior
- best-aligned head ablation summaries

### `Figures/DiT_T5_the_repr_bias/`

Prompt-bias and representation-bias outputs:
- `T5_shape12_the_repr_bias_spatial_sim_df.csv`
- `T5_shape12_wordvecs_original.pkl`
- `T5_shape12_wordvecs_the1.pkl`
- `T5_shape12_wordvecs_the2.pkl`
- `T5_shape12_wordvecs_the12.pkl`
- `objrel_T5_DiT_mini_pilot_L1H2_pos_embed_inprod_with_spatial_rel_proj_original.png`
- `objrel_T5_DiT_mini_pilot_L1H2_pos_embed_inprod_with_spatial_rel_proj_original.pdf`
- `objrel_T5_DiT_mini_pilot_L1H2_pos_embed_inprod_with_spatial_rel_proj_original_align_df.csv`
- `objrel_T5_DiT_mini_pilot_L1H2_pos_embed_inprod_with_spatial_rel_proj_the2_vecperturb.png`
- `objrel_T5_DiT_mini_pilot_L1H2_pos_embed_inprod_with_spatial_rel_proj_the2_vecperturb.pdf`
- `objrel_T5_DiT_mini_pilot_L1H2_pos_embed_inprod_with_spatial_rel_proj_the2_vecperturb_align_df.csv`
- `objrel_T5_DiT_mini_pilot_L2H3_pos_embed_inprod_with_spatial_rel_proj_original.png`
- `objrel_T5_DiT_mini_pilot_L2H3_pos_embed_inprod_with_spatial_rel_proj_original.pdf`
- `objrel_T5_DiT_mini_pilot_L2H3_pos_embed_inprod_with_spatial_rel_proj_original_align_df.csv`

These artifacts support the analysis of:
- whether adding `the` changes token geometry
- how relation-sensitive heads interact with positional embeddings
- how vector perturbations affect relation projections

### `Figures/model_eval_synopsis/`

Model-evaluation summary figures:
- `hk_objrel_singleobj_T5_mini_pilot1_eval_train_dynamics_traj_syn.png`
- `hk_objrel_singleobj_T5_mini_pilot1_eval_train_dynamics_traj_syn.pdf`
- `hk_objrel_singleobj_T5_mini_pilot1_eval_train_dynamics_traj_syn.svg`
- `hk_objrel_singleobj_mini_rndemb_eval_train_dynamics_traj_syn.png`
- `hk_objrel_singleobj_mini_rndemb_eval_train_dynamics_traj_syn.pdf`
- `hk_objrel_singleobj_mini_rndemb_eval_train_dynamics_traj_syn.svg`

These summarize training dynamics and evaluation trajectories for T5 and random-embedding variants.

## Results Generated Inside the Notebooks

Not every notebook result is exported as a standalone file. Several important outputs are generated interactively inside the notebooks and may only live in notebook cells unless explicitly saved.

Examples include:
- OV-QK downstream candidate tables
- write-aware ranking comparison tables
- method-winner summaries for alternate ranking rules
- slot-specific chain score heatmaps
- read-vs-write diagnostic scatters
- variance-partition result tables
- compensation and perturbation heatmaps
- qualitative baseline vs ablation image grids
- attention-map diagnostics

In particular, the current `02_ablation_and_causality.ipynb` adds:
- an early OV-QK-OV candidate-discovery section before heavy ablations
- richer circuit-diagnostic plots
- a ranking-comparison section with stricter write-aware alternatives
- a stop cell so the notebook can halt before later ablation blocks

## Recommended Reading Order

If you are new to the project, the cleanest order is:

1. `notebooks/split_workflows/01_setup_and_head_discovery.ipynb`
2. `notebooks/split_workflows/02_ablation_and_causality.ipynb`
3. `notebooks/split_workflows/03_identifiability_and_variance_partition.ipynb`
4. `notebooks/split_workflows/04_prompt_bias_and_attention_maps.ipynb`

Then consult:
- `notebooks/checkpoint_training_trajectory_analysis.ipynb` for checkpoint-scale training dynamics
- `notebooks/20260126_T5_word_embed_var_factorization_head_finding_MINI.ipynb` for the original all-in-one notebook

## Quick Summary of What Has Been Added

New or newly split analysis entry points:
- `01_setup_and_head_discovery.ipynb`
- `02_ablation_and_causality.ipynb`
- `03_identifiability_and_variance_partition.ipynb`
- `04_prompt_bias_and_attention_maps.ipynb`

Key notebook-support modules added or used to support that split:
- `downstream_head_tracing.py`
- `variance_partition_with_effects.py`
- `pixart_utils.py`
- `pixart_sampling_utils.py`
- `zero_head_ablation_utils.py`
- `eval_cached_embeddings.py`
- `relation_shape_dataset_lib.py`
- `text_encoder_control_lib.py`
- `image_utils.py`
- `attention_map_store_utils.py`
- `pixart_pos_embed.py`

Saved result families currently present:
- training checkpoints
- cached T5 embeddings
- per-checkpoint generalization evaluation tables
- per-checkpoint object detection pickles
- training-curve plots
- head alignment synopses
- ablation sweep figures
- representation-bias figures
- model-evaluation summary plots

