# Awesome Unified Embedding [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated list of papers, models, datasets, and benchmarks for **unified multi-modal embedding models** — from CLIP to Gemini Embedding 2.


**Last updated**: 2026-04 · **References**: 500+ 


## Contents

- [Contents](#contents)
- [Industry frontier (2024–2026)](#industry-frontier-20242026)
- [Dual-encoder vision-language (CLIP family)](#dual-encoder-vision-language-clip-family)
  - [Foundational](#foundational)
  - [Scaling \& open reproductions](#scaling--open-reproductions)
  - [Sigmoid \& shape-optimized](#sigmoid--shape-optimized)
  - [Multilingual \& Chinese CLIPs](#multilingual--chinese-clips)
  - [Efficient variants](#efficient-variants)
  - [Region-aware \& dense](#region-aware--dense)
  - [Long-context](#long-context)
  - [Re-caption enhanced](#re-caption-enhanced)
  - [Improved recipes \& unified encoders](#improved-recipes--unified-encoders)
  - [Hard-negative / compositional](#hard-negative--compositional)
  - [Video / multilingual / high-resolution](#video--multilingual--high-resolution)
- [LLM / MLLM-as-embedder](#llm--mllm-as-embedder)
  - [Text-only LLM embedders](#text-only-llm-embedders)
  - [MLLM-based multimodal embedders](#mllm-based-multimodal-embedders)
  - [Document / visual retrieval (ColPali family)](#document--visual-retrieval-colpali-family)
  - [Reasoning-intensive retrieval](#reasoning-intensive-retrieval)
  - [Late-interaction / multi-vector](#late-interaction--multi-vector)
- [Omni-modal / Any-to-Any](#omni-modal--any-to-any)
  - [Binding approaches](#binding-approaches)
  - [Native joint \& tokenized](#native-joint--tokenized)
  - [Audio-language](#audio-language)
  - [Video-language](#video-language)
  - [3D / point-cloud](#3d--point-cloud)
- [Domain-specialised embedders](#domain-specialised-embedders)
  - [Biomedical](#biomedical)
  - [Remote sensing / geospatial](#remote-sensing--geospatial)
- [Training techniques](#training-techniques)
  - [Contrastive objectives](#contrastive-objectives)
  - [Hard-negative mining](#hard-negative-mining)
  - [Instruction / task conditioning](#instruction--task-conditioning)
  - [Synthetic data generation](#synthetic-data-generation)
  - [Distillation](#distillation)
  - [Matryoshka \& adaptive dim](#matryoshka--adaptive-dim)
  - [Pooling \& causal→bidirectional conversion](#pooling--causalbidirectional-conversion)
  - [PEFT / LoRA](#peft--lora)
- [Benchmarks](#benchmarks)
  - [Text](#text)
  - [Multimodal](#multimodal)
  - [Video / audio / 3D](#video--audio--3d)
  - [Compositionality \& robustness](#compositionality--robustness)
- [Datasets (training)](#datasets-training)
- [Surveys \& positional papers](#surveys--positional-papers)
- [License](#license)

---

## Industry frontier (2024–2026)

| Model | Release | Modalities | Max ctx | Langs | MTEB / MMEB | Openness |
|---|---|---|---|---|---|---|
| **Gemini Embedding 2** (Google DeepMind) | 2026-03 | text, image, video, audio, docs | 8192 text; 120s video; 6 img / 6-pg PDF | 100+ | MTEB-Eng 68.32; MMEB 68.9 | Closed |
| **Qwen3-VL-Embedding-8B** (Alibaba) ★ | 2026-01 | text, image, video, doc | 32K | 30+ | **MMEB-V2 77.8 (#1)** | Open |
| **Cohere Embed v4** | 2025-04 | text, image, interleaved | 128K | 100+ | SOTA agentic | Closed |
| **Voyage voyage-3-large** | 2025-01 | text | 32K | multi | +9.74% vs OAI-3L | Closed |
| **Voyage voyage-multimodal-3** | 2024-11 | text + interleaved image | — | EN | +41% vs OAI-CLIP | Closed |
| **Voyage voyage-code-3** | 2024-12 | text + code | — | code | SOTA code | Closed |
| **Mistral Codestral Embed** | 2025-05 | text + code | — | code | beats voyage-code-3 | Closed |
| **Gemini Embedding 001** (Google) | 2025-03 | text | 2048 | 100+ | MTEB-Multi 68.32 | Closed |
| **EmbeddingGemma-300M** (Google) ★ | 2025-09 | text | 2048 | 100+ | #1 open <500M | Open |
| **OpenAI text-embedding-3-large** | 2024-01 | text | 8191 | multi | MTEB 64.6 | Closed |
| **Jina jina-embeddings-v4** ★ | 2025-06 | text + image + interleaved | 32K | multi | ViDoRe SOTA | Open |
| **Jina jina-embeddings-v3** ★ | 2024-09 | text | 8192 | 30+ | beats OAI/Cohere peers | Open |
| **Nomic Embed v2-MoE** ★ | 2025-02 | text | — | multi | SOTA open-MoE | Open |
| **Nomic Embed Multimodal 7B** ★ | 2025 | text + image + PDF | — | multi | ViDoRe-v2 62.7 | Open |
| **Qwen3-Embedding 0.6B/4B/8B** ★ | 2025-06 | text | 32K | multi | MMTEB 70.58 | Open |
| **BGE-M3** (BAAI) ★ | 2024-02 | text (dense+sparse+MV) | 8192 | 170+ | MIRACL SOTA | Open |
| **NV-Embed v2** (NVIDIA) ★ | 2024-08 | text | 32K | EN | MTEB 72.31 | Open |
| **MM-Embed** (NVIDIA) ★ | 2024-11 | text + image | — | EN | UniIR 52.7 | Open |
| **Arctic Embed 2.0** (Snowflake) | 2024-12 | text | — | multi | competitive | Open |
| **SFR-Embedding-Mistral/2R** | 2024 | text | 32K | EN | MTEB 67.6 / 70.3 | Open |
| **Linq-Embed-Mistral** | 2024 | text | 32K | EN | MTEB 68.2 | Open |
| **Jasper + Stella** (NovaSearch) ★ | 2024-12 | text + image | — | EN | MTEB 71.54 | Open |

---

## Dual-encoder vision-language (CLIP family)

### Foundational

- **CLIP** (OpenAI, 2021) — `arXiv:2103.00020` · [paper](https://arxiv.org/abs/2103.00020) · [code](https://github.com/openai/CLIP) — Image-text InfoNCE on 400M WIT pairs; zero-shot ImageNet 76.2%.
- **ALIGN** (Google, 2021) — `arXiv:2102.05918` · [paper](https://arxiv.org/abs/2102.05918) — 1.8B noisy alt-text; EfficientNet-L2 + BERT-Large dual encoder.
- **FILIP** (Huawei, 2022) — `arXiv:2111.07783` · [paper](https://arxiv.org/abs/2111.07783) — Token-wise max-sim late interaction.
- **DeCLIP** (SenseTime, 2022) — `arXiv:2110.05208` · [paper](https://arxiv.org/abs/2110.05208) — Multi-source supervision; 7× data efficiency.
- **SLIP** (Meta, 2022) — `arXiv:2112.12750` · [paper](https://arxiv.org/abs/2112.12750) — CLIP + SimCLR multi-task.

### Scaling & open reproductions

- **LAION-400M** (LAION, 2021) — `arXiv:2111.02114` · [paper](https://arxiv.org/abs/2111.02114) — First open 400M image-text corpus.
- **LAION-5B** (LAION, 2022) — `arXiv:2210.08402` · [paper](https://arxiv.org/abs/2210.08402) — 5.85B pairs (2.32B EN).
- **OpenCLIP scaling laws** (LAION, 2023) — `arXiv:2212.07143` · [paper](https://arxiv.org/abs/2212.07143) · [code](https://github.com/mlfoundations/open_clip) — ★ Reproducible power-law scaling on LAION.
- **DataComp** (U.Wash, 2023) — `arXiv:2304.14108` · [paper](https://arxiv.org/abs/2304.14108) · [code](https://github.com/mlfoundations/datacomp) — 12.8B candidate pool data benchmark.
- **MetaCLIP** (Meta, 2024) — `arXiv:2309.16671` · [paper](https://arxiv.org/abs/2309.16671) · [code](https://github.com/facebookresearch/MetaCLIP) — Metadata-curated 400M / 2.5B; ViT-H 80.5% IN-1K.
- **MetaCLIP 2** (Meta, 2025) — `arXiv:2507.22062` · [paper](https://arxiv.org/abs/2507.22062) — Worldwide 300+ lang recipe; cures curse of multilinguality.
- **DFN** (Apple/UW, 2024) — `arXiv:2309.17425` · [paper](https://arxiv.org/abs/2309.17425) — Data Filtering Networks; DFN-5B ViT-H@378 = 84.4% IN-1K.
- **DataComp-LM** (U.Wash, 2024) — `arXiv:2406.11794` · [paper](https://arxiv.org/abs/2406.11794) — Sibling text-only 240T-token corpus.

### Sigmoid & shape-optimized

- **SigLIP** (Google, 2023) — `arXiv:2303.15343` · [paper](https://arxiv.org/abs/2303.15343) · [code](https://github.com/google-research/big_vision) — Pairwise sigmoid loss; small-batch-friendly.
- **SoViT / SigLIP-SO400m** (Google, 2023) — `arXiv:2305.13035` · [paper](https://arxiv.org/abs/2305.13035) — Shape-optimised 400M backbone; de-facto LLaVA/PaliGemma vision tower.
- **SigLIP 2** (DeepMind, 2025) ★ — `arXiv:2502.14786` · [paper](https://arxiv.org/abs/2502.14786) — Captioning + self-distill + masked pred + NaFlex; SOTA at every scale.

### Multilingual & Chinese CLIPs

- **Chinese-CLIP** (Alibaba, 2022) — `arXiv:2211.01335` · [paper](https://arxiv.org/abs/2211.01335) · [code](https://github.com/OFA-Sys/Chinese-CLIP) — Two-stage zh/en training.
- **AltCLIP** (BAAI, 2022) — `arXiv:2211.06679` · [paper](https://arxiv.org/abs/2211.06679) — Swap CLIP text with XLM-R; teacher+contrastive.
- **mCLIP** (2023, ACL) — [paper](https://aclanthology.org/2023.acl-long.728/) — Triangle cross-lingual distillation.
- **NLLB-CLIP** (2023) — `arXiv:2309.01859` · [paper](https://arxiv.org/abs/2309.01859) — 201 languages on \$1K budget.
- **Jina-CLIP v1** (Jina, 2024) — `arXiv:2405.20204` · [paper](https://arxiv.org/abs/2405.20204) — Multi-task (text-text + image-text) → CLIP also works as text retriever.
- **Jina-CLIP v2** (Jina, 2024) ★ — `arXiv:2412.08802` · [paper](https://arxiv.org/abs/2412.08802) · [model](https://huggingface.co/jinaai/jina-clip-v2) — 89 langs, 512×512, MRL 64–1024.

### Efficient variants

- **LiT** (Google, 2022) — `arXiv:2111.07991` · [paper](https://arxiv.org/abs/2111.07991) — Freeze vision; tune only text; ViT-g LiT = 85.2% IN-1K.
- **EVA-CLIP** (BAAI, 2023) — `arXiv:2303.15389` · [paper](https://arxiv.org/abs/2303.15389) · [code](https://github.com/baaivision/EVA/tree/master/EVA-CLIP) — MIM init + LAMB.
- **EVA-02** (BAAI, 2023) — `arXiv:2303.11331` · [paper](https://arxiv.org/abs/2303.11331) — Backbone used by EVA-CLIP.
- **EVA-CLIP-18B** (BAAI, 2024) — `arXiv:2402.04252` · [paper](https://arxiv.org/abs/2402.04252) — 18B params; 80.7% avg / 27 cls benchmarks.
- **TinyCLIP** (Microsoft, 2023) — `arXiv:2309.12314` · [paper](https://arxiv.org/abs/2309.12314) — Affinity mimicking + weight inheritance.
- **MobileCLIP** (Apple, 2024) — `arXiv:2311.17049` · [paper](https://arxiv.org/abs/2311.17049) — Hybrid CNN-transformer + reinforced data.
- **MobileCLIP2** (Apple, 2025) — `arXiv:2508.20691` · [paper](https://arxiv.org/abs/2508.20691) — DFNDR-2B dataset; matches SigLIP-SO400m.
- **FLIP** (Meta, 2023) — `arXiv:2212.00794` · [paper](https://arxiv.org/abs/2212.00794) — 50% patch masking; 2× speedup.
- **CLIPA** (UCSC, 2023) — `arXiv:2305.07017` · [paper](https://arxiv.org/abs/2305.07017) — Inverse scaling law for CLIP.
- **CLIPA-v2** (UCSC, 2023) — `arXiv:2306.15658` · [paper](https://arxiv.org/abs/2306.15658) — ViT-H/14 → 81.1% IN-1K on \$10K budget.
- **RECLIP** (2023) — `arXiv:2304.06028` · [paper](https://arxiv.org/abs/2304.06028) — Small-image pretraining; 6× savings.

### Region-aware & dense

- **RegionCLIP** (Microsoft, 2022) — `arXiv:2112.09106` · [paper](https://arxiv.org/abs/2112.09106).
- **GLIP** (Microsoft, 2022) — `arXiv:2112.03857` · [paper](https://arxiv.org/abs/2112.03857) — Unifies detection + phrase grounding.
- **MaskCLIP** (Microsoft, 2023) — `arXiv:2208.12262` · [paper](https://arxiv.org/abs/2208.12262) — Masked self-distillation.
- **CLIPSelf** (2024) — `arXiv:2310.01403` · [paper](https://arxiv.org/abs/2310.01403) — Self-distillation w/o region-text.
- **Alpha-CLIP** (2024) — `arXiv:2312.03818` · [paper](https://arxiv.org/abs/2312.03818) — α-channel input for region emphasis.
- **CLOC** (Apple, 2024) — `arXiv:2410.02746` · [paper](https://arxiv.org/abs/2410.02746) — Promptable embeddings for MLLM region tasks.
- **SPARC** (DeepMind, 2024) — `arXiv:2401.09865` · [paper](https://arxiv.org/abs/2401.09865) — Sparse patch-token grouping.
- **PyramidCLIP** (NeurIPS, 2022) — `arXiv:2204.14095` · [paper](https://arxiv.org/abs/2204.14095).

### Long-context

- **Long-CLIP** (ECCV, 2024) — `arXiv:2403.15378` · [paper](https://arxiv.org/abs/2403.15378) — 77 → 248 tokens.
- **LoTLIP** (NeurIPS, 2024) — `arXiv:2410.05249` · [paper](https://arxiv.org/abs/2410.05249) — Corner-token aggregation.
- **DCI** (Meta, 2024) — `arXiv:2312.08578` · [paper](https://arxiv.org/abs/2312.08578) — 1000+ word SA-1B human captions.

### Re-caption enhanced

- **LaCLIP** (NeurIPS, 2023) — `arXiv:2305.20088` · [paper](https://arxiv.org/abs/2305.20088) — LLaMA caption rewrites.
- **VeCLIP** (Apple, 2024) — `arXiv:2310.07699` · [paper](https://arxiv.org/abs/2310.07699) — Visual-enriched captions.
- **DreamLIP** (2024) — `arXiv:2403.17007` · [paper](https://arxiv.org/abs/2403.17007) — Long MLLM captions + sub-caption sampling.
- **Recap-DataComp-1B** (ICML, 2025) — `arXiv:2406.08478` · [paper](https://arxiv.org/abs/2406.08478) — 1.3B images re-captioned by LLaMA-3-LLaVA.
- **ShareGPT4V** (ECCV, 2024) — `arXiv:2311.12793` · [paper](https://arxiv.org/abs/2311.12793) — 1.2M GPT4V captions.
- **CLIPS** (UCSC, 2024) — `arXiv:2411.16828` · [paper](https://arxiv.org/abs/2411.16828) — Short input + long-synthetic target.
- **LLM2CLIP** (Microsoft, 2024) — `arXiv:2411.04997` · [paper](https://arxiv.org/abs/2411.04997) — LLaMA-3 text encoder into CLIP.
- **HQ-CLIP** (2025) — `arXiv:2507.22431` · [paper](https://arxiv.org/abs/2507.22431) — VLM-150M dataset.

### Improved recipes & unified encoders

- **CoCa** (Google, 2022) — `arXiv:2205.01917` · [paper](https://arxiv.org/abs/2205.01917) — Contrastive + captioning; 86.3% IN-1K.
- **FLAVA** (Meta, 2022) — `arXiv:2112.04482` · [paper](https://arxiv.org/abs/2112.04482).
- **ALBEF** (Salesforce, 2021) — `arXiv:2107.07651` · [paper](https://arxiv.org/abs/2107.07651) — ITC + momentum distillation.
- **BLIP** (Salesforce, 2022) — `arXiv:2201.12086` · [paper](https://arxiv.org/abs/2201.12086).
- **BLIP-2** (Salesforce, 2023) — `arXiv:2301.12597` · [paper](https://arxiv.org/abs/2301.12597) — Q-Former bridge.
- **BEiT-3** (Microsoft, 2023) — `arXiv:2208.10442` · [paper](https://arxiv.org/abs/2208.10442).
- **BridgeTower** (AAAI, 2023) — `arXiv:2206.08657` · [paper](https://arxiv.org/abs/2206.08657).
- **UniCL** (CVPR, 2022) — `arXiv:2204.03610` · [paper](https://arxiv.org/abs/2204.03610).
- **K-LITE** (NeurIPS, 2022) — `arXiv:2204.09222` · [paper](https://arxiv.org/abs/2204.09222).
- **Florence** (Microsoft, 2021) — `arXiv:2111.11432` · [paper](https://arxiv.org/abs/2111.11432).
- **Florence-2** (Microsoft, 2024) — `arXiv:2311.06242` · [paper](https://arxiv.org/abs/2311.06242).
- **CLOOB** (NeurIPS, 2022) — `arXiv:2110.11316` · [paper](https://arxiv.org/abs/2110.11316) — Hopfield + InfoLOOB.
- **STAIR** (Apple, 2023) — `arXiv:2301.13081` · [paper](https://arxiv.org/abs/2301.13081) — Sparse interpretable tokens.
- **Nomic Embed Vision** (Nomic, 2024) ★ — `arXiv:2406.18587` · [paper](https://arxiv.org/abs/2406.18587) — Unified text+image latent vs OpenAI CLIP.
- **Perception Encoder** (Meta, 2025) ★ — `arXiv:2504.13181` · [paper](https://arxiv.org/abs/2504.13181) — Best visual embeddings at intermediate layers.
- **OpenVision** (UCSC, 2025) — `arXiv:2505.04601` · [paper](https://arxiv.org/abs/2505.04601) — Tiny→Huge open vision encoders.
- **OpenVision 2** (UCSC, 2026) — `arXiv:2509.01644` · [paper](https://arxiv.org/abs/2509.01644) — Captioning-only, 1.5× faster.
- **TULIP** (Berkeley, 2025) — `arXiv:2503.15485` · [paper](https://arxiv.org/abs/2503.15485) — Generative aug + II/TT contrastive.
- **PerceptionCLIP** (ICLR, 2024) — `arXiv:2308.01313` · [paper](https://arxiv.org/abs/2308.01313) — Training-free context conditioning.
- **DIVA** (BAAI, 2025) — `arXiv:2407.20171` · [paper](https://arxiv.org/abs/2407.20171) — Diffusion feedback.
- **NaViT** (Google, 2023) — `arXiv:2307.06304` · [paper](https://arxiv.org/abs/2307.06304) — Native-resolution patch packing.

### Hard-negative / compositional

- **NegCLIP / ARO** (ICLR, 2023) — `arXiv:2210.01936` · [paper](https://arxiv.org/abs/2210.01936) — Word-swap hard negatives.
- **SugarCrepe** (NeurIPS, 2023) — `arXiv:2306.14610` · [paper](https://arxiv.org/abs/2306.14610) — Fixes hackable benchmarks.
- **TripletCLIP** (NeurIPS, 2024) — `arXiv:2411.02545` · [paper](https://arxiv.org/abs/2411.02545) — LLM + T2I negatives.
- **CE-CLIP** (CVPR 2024) — `arXiv:2306.08832` · [paper](https://arxiv.org/abs/2306.08832) — Intra-modal contrast + cross-modal ranking HN.
- **IL-CLIP** (CVPR 2024) — Iterated learning fine-tune for compositionality.
- **CLoVe** (2024) — `arXiv:2402.15021` · [paper](https://arxiv.org/abs/2402.15021) — Data-curation + patching recipe.
- **VisMin** (NeurIPS 2024) — Minimal-change image-pair benchmark *and* training set.
- **NegBench** (CVPR 2025) — `arXiv:2501.09425` · [paper](https://arxiv.org/abs/2501.09425) — 79K negation-stress benchmark + CC-Neg.

### Modality-gap geometry & self-distillation

- **AlignCLIP** (ICLR 2025) — [OpenReview](https://openreview.net/forum?id=aPTGvFqile) — Shared-param encoder closes modality gap.
- **CLIP-Refine** (2025) — `arXiv:2504.12717` · [paper](https://arxiv.org/abs/2504.12717) — One-epoch RaFA + HyCD post-pretraining.
- **Ex-MCR** (NeurIPS 2024) — Paired-data-free merge of bi-modal CLIP spaces.
- **QUEST** (NeurIPS 2024) — Quadruple multimodal contrastive with modality-gap constraints.
- **SILC** (ECCV 2024) — Local-to-global EMA self-distillation on CLIP.
- **CLIP-KD** (CVPR 2024) — Empirical distillation recipes for small CLIPs.
- **DCLIP** (2025) — `arXiv:2505.21549` · [paper](https://arxiv.org/abs/2505.21549) — Meta-teacher + region-attention distillation.
- **COSMOS** (CVPR 2025) — Cross-modality EMA self-distill (CLIP + DINO).
- **ViTamin** (CVPR 2024) — `arXiv:2404.02132` · [paper](https://arxiv.org/abs/2404.02132) — Hybrid ViT-CNN backbone for VL.
- **UNIT** (NeurIPS 2024) — Unified image + text recognition encoder.
- **ELIP** (2025) — `arXiv:2502.15682` · [paper](https://arxiv.org/abs/2502.15682) — Text-conditional visual prompts on frozen CLIP/SigLIP/BLIP-2.

### Self-supervised vision foundations as embedders

- **DINOv3** (Meta, 2025) ★ — `arXiv:2508.10104` · [paper](https://arxiv.org/abs/2508.10104) — 7B SSL backbone; Gram anchoring; SOTA dense retrieval w/o text.
- **V-JEPA** (Meta, 2024) — `arXiv:2404.08471` · [paper](https://arxiv.org/abs/2404.08471) — Feature-prediction SSL for video.
- **AIMv2** (Apple, 2024) — `arXiv:2411.14402` · [paper](https://arxiv.org/abs/2411.14402) — Autoregressive image+text pretraining.

### Synthetic-caption / data-centric

- **CapsFusion** (CVPR 2024) — `arXiv:2310.20550` · [paper](https://arxiv.org/abs/2310.20550) — Fuses alt-text + synthetic captions.
- **SynthCLIP** (2024) — `arXiv:2402.01832` · [paper](https://arxiv.org/abs/2402.01832) — Fully-synthetic CLIP training counterpoint.
- **Revisit-DataComp** (2024) — `arXiv:2410.02740` · [paper](https://arxiv.org/abs/2410.02740) — Caption-mix study.

### Video / multilingual / high-resolution

- **InternVL** (OpenGVLab, 2024) — `arXiv:2312.14238` · [paper](https://arxiv.org/abs/2312.14238) — 6B ViT + 8B middleware.
- **PaLI** (Google, 2023) — `arXiv:2209.06794` · [paper](https://arxiv.org/abs/2209.06794) — WebLI 10B/109 langs.
- **PaLI-3** (Google, 2023) — `arXiv:2310.09199` · [paper](https://arxiv.org/abs/2310.09199) — 2B SigLIP + 5B full.
- **CLIP-ViP** (2022) — `arXiv:2209.06430` · [paper](https://arxiv.org/abs/2209.06430).
- **InternVid / ViCLIP** (ICLR, 2024) — `arXiv:2307.06942` · [paper](https://arxiv.org/abs/2307.06942) — 234M clips.
- **X-CLIP** (ACM MM, 2022) — `arXiv:2207.07285` · [paper](https://arxiv.org/abs/2207.07285).

---

## LLM / MLLM-as-embedder

### Text-only LLM embedders

- **RepLLaMA** (2023) — `arXiv:2310.08319` · [paper](https://arxiv.org/abs/2310.08319) — LLaMA-2-7B + EOS pooling.
- **Llama2Vec** (2023) — `arXiv:2312.15503` · [paper](https://arxiv.org/abs/2312.15503) — EBAE/EBAR unsupervised.
- **E5-Mistral-7B-Instruct** (Microsoft, 2024) ★ — `arXiv:2401.00368` · [paper](https://arxiv.org/abs/2401.00368) · [model](https://huggingface.co/intfloat/e5-mistral-7b-instruct) — 500K synthetic + instruction prefix.
- **Multilingual E5** (Microsoft, 2024) — `arXiv:2402.05672` · [paper](https://arxiv.org/abs/2402.05672).
- **Echo Embeddings** (CMU, 2024) — `arXiv:2402.15449` · [paper](https://arxiv.org/abs/2402.15449) — Input repetition trick.
- **LLM2Vec** (McGill, 2024) ★ — `arXiv:2404.05961` · [paper](https://arxiv.org/abs/2404.05961) · [code](https://github.com/McGill-NLP/llm2vec) — Bidir + MNTP + SimCSE recipe.
- **GritLM** (Contextual AI, 2024) — `arXiv:2402.09906` · [paper](https://arxiv.org/abs/2402.09906) · [code](https://github.com/ContextualAI/gritlm) — Unified gen+embed.
- **NV-Embed v1/v2** (NVIDIA, 2024) ★ — `arXiv:2405.17428` · [paper](https://arxiv.org/abs/2405.17428) — Latent-attention pooling; MTEB 72.31.
- **NV-Retriever** (NVIDIA, 2024) — `arXiv:2407.15831` · [paper](https://arxiv.org/abs/2407.15831) — Positive-aware hard-neg mining.
- **SFR-Embedding-Mistral / 2_R** (Salesforce, 2024) — [blog](https://www.salesforce.com/blog/sfr-embedding/).
- **Linq-Embed-Mistral** (2024) — `arXiv:2412.03223` · [paper](https://arxiv.org/abs/2412.03223).
- **gte-Qwen2-7B-instruct** (Alibaba, 2024) — [series arXiv:2308.03281](https://arxiv.org/abs/2308.03281) · [model](https://huggingface.co/Alibaba-NLP/gte-Qwen2-7B-instruct) — MTEB 70.24.
- **BGE-M3** (BAAI, 2024) ★ — `arXiv:2402.03216` · [paper](https://arxiv.org/abs/2402.03216) · [code](https://github.com/FlagOpen/FlagEmbedding) — Dense + sparse + MV multilingual.
- **bge-en-icl** (BAAI, 2024) — `arXiv:2409.15700` · [paper](https://arxiv.org/abs/2409.15700) — In-context examples.
- **Instructor** (2023) — `arXiv:2212.09741` · [paper](https://arxiv.org/abs/2212.09741) · [code](https://github.com/xlang-ai/instructor-embedding).
- **TART** (2022) — `arXiv:2211.09260` · [paper](https://arxiv.org/abs/2211.09260).
- **Promptriever** (2024) — `arXiv:2409.11136` · [paper](https://arxiv.org/abs/2409.11136).
- **GISTEmbed** (2024) — `arXiv:2402.16829` · [paper](https://arxiv.org/abs/2402.16829) — Guide-model false-neg filtering.
- **CDE** (2024) — `arXiv:2410.02525` · [paper](https://arxiv.org/abs/2410.02525) — Contextual batching.
- **Pooling-and-Attention** (2024) — `arXiv:2409.02727` · [paper](https://arxiv.org/abs/2409.02727).
- **Nomic Embed v1** (Nomic, 2024) — `arXiv:2402.01613` · [paper](https://arxiv.org/abs/2402.01613).
- **Nomic Embed v2-MoE** (Nomic, 2025) ★ — [HF](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe).
- **Arctic-Embed** (Snowflake, 2024) — `arXiv:2405.05374` · [paper](https://arxiv.org/abs/2405.05374).
- **Arctic-Embed 2.0** (Snowflake, 2024) — `arXiv:2412.04506` · [paper](https://arxiv.org/abs/2412.04506).
- **jina-embeddings-v3** (Jina, 2024) — `arXiv:2409.10173` · [paper](https://arxiv.org/abs/2409.10173) — Task-LoRA adapters.
- **Jasper + Stella** (NovaSearch, 2024) — `arXiv:2412.19048` · [paper](https://arxiv.org/abs/2412.19048) — Multi-teacher distillation.
- **Causal2Vec** (2025) — `arXiv:2507.23386` · [paper](https://arxiv.org/abs/2507.23386) — Prepended contextual token.
- **Qwen3-Embedding** (Alibaba, 2025) ★ — `arXiv:2506.05176` · [paper](https://arxiv.org/abs/2506.05176).
- **EmbeddingGemma** (Google, 2025) ★ — `arXiv:2509.20354` · [paper](https://arxiv.org/abs/2509.20354) — 308M SOTA open <500M.
- **Gemini Embedding** (Google, 2025) — `arXiv:2503.07891` · [paper](https://arxiv.org/abs/2503.07891).
- **Late Chunking** (Jina, 2024) — `arXiv:2409.04701` · [paper](https://arxiv.org/abs/2409.04701) — Embed long doc whole, chunk after pool.
- **LongEmbed (extension methods)** (EMNLP 2024) — Plug-and-play PI/RoPE/NTK for long ctx.
- **AutoRegEmbed** (EMNLP 2025) — Compress input via AR conditional probability.
- **ExpandR** (EMNLP 2025) — Joint retriever + LLM optimisation w/ ranking-pref alignment.
- **PromptReps** (EMNLP 2024) — `arXiv:2404.18424` · [paper](https://arxiv.org/abs/2404.18424) — Prompt-only LLM as joint dense+sparse, no training.
- **EntailmentTuning** (EMNLP 2024) — Entailment-style fine-tune for DPR.
- **Mistral-SPLADE** (2024) — `arXiv:2408.11119` · [paper](https://arxiv.org/abs/2408.11119) — LLM-backbone SPLADE; dense/sparse boundary.
- **Li-LSR** (2025) — `arXiv:2505.01452` · [paper](https://arxiv.org/abs/2505.01452) — Inference-free retrieval for sparse reps.
- **Llama-Embed-Nemotron-8B** (NVIDIA, 2025) — Llama-backbone generalist text embedder.

### MLLM-based multimodal embedders

- **UniIR** (ECCV, 2024) — `arXiv:2311.17136` · [paper](https://arxiv.org/abs/2311.17136) · [project](https://tiger-ai-lab.github.io/UniIR/) — Introduces M-BEIR; 10 datasets × 8 tasks.
- **PreFLMR / FLMR** (ACL, 2024) — `arXiv:2402.08327` · [paper](https://arxiv.org/abs/2402.08327) — BLIP-2 late-interaction.
- **MagicLens** (ICML, 2024) — `arXiv:2403.19651` · [paper](https://arxiv.org/abs/2403.19651) — 36.7M web-mined triplets.
- **MARVEL** (ACL, 2024) — `arXiv:2310.14037` · [paper](https://arxiv.org/abs/2310.14037).
- **UniVL-DR** (ICLR, 2023) — `arXiv:2209.00179` · [paper](https://arxiv.org/abs/2209.00179).
- **DSE** (EMNLP, 2024) — `arXiv:2406.11251` · [paper](https://arxiv.org/abs/2406.11251) — Doc-Screenshot Embedding.
- **VisRAG** (2024) — `arXiv:2410.10594` · [paper](https://arxiv.org/abs/2410.10594) — VLM RAG on doc images.
- **VISTA** (ACL, 2024) — `arXiv:2406.04292` · [paper](https://arxiv.org/abs/2406.04292).
- **E5-V** (2024) — `arXiv:2407.12580` · [paper](https://arxiv.org/abs/2407.12580) — Text-only training of MLLM embedder.
- **VLM2Vec** (ICLR, 2025) ★ — `arXiv:2410.05160` · [paper](https://arxiv.org/abs/2410.05160) · [code](https://github.com/TIGER-AI-Lab/VLM2Vec) — Introduces MMEB.
- **VLM2Vec-V2** (2025) ★ — `arXiv:2507.04590` · [paper](https://arxiv.org/abs/2507.04590) — MMEB-V2 with video/doc.
- **MM-Embed** (NVIDIA, ICLR 2025) ★ — `arXiv:2411.02571` · [paper](https://arxiv.org/abs/2411.02571) — UniIR 52.7.
- **LamRA** (CVPR, 2025) — `arXiv:2412.01720` · [paper](https://arxiv.org/abs/2412.01720).
- **GME** (Alibaba, 2024) ★ — `arXiv:2412.16855` · [paper](https://arxiv.org/abs/2412.16855) — Qwen2-VL-based UMRB SOTA.
- **MegaPairs / MMRet / BGE-VL** (BAAI, 2024) ★ — `arXiv:2412.14475` · [paper](https://arxiv.org/abs/2412.14475) — 26M synthetic triplets.
- **mmE5** (2025) — `arXiv:2502.08468` · [paper](https://arxiv.org/abs/2502.08468) — Multilingual multimodal deep-thinking synthetic.
- **LLaVE** (2025) — `arXiv:2503.04812` · [paper](https://arxiv.org/abs/2503.04812) — Hardness-weighted; +6.2 over SOTA.
- **UniME** (ACM MM, 2025) — `arXiv:2504.17432` · [paper](https://arxiv.org/abs/2504.17432).
- **UniME-V2** (2025) — `arXiv:2510.13515` · [paper](https://arxiv.org/abs/2510.13515).
- **U-MARVEL** (2025) — `arXiv:2507.14902` · [paper](https://arxiv.org/abs/2507.14902).
- **jina-embeddings-v4** (Jina, 2025) ★ — `arXiv:2506.18902` · [paper](https://arxiv.org/abs/2506.18902) — Qwen2.5-VL-3B + MV.
- **Omni-Embed-Nemotron** (NVIDIA, 2025) — `arXiv:2510.03458` · [paper](https://arxiv.org/abs/2510.03458) — + audio + video.
- **Qwen3-VL-Embedding** (Alibaba, 2026) ★ — `arXiv:2601.04720` · [paper](https://arxiv.org/abs/2601.04720) — MMEB-V2 77.8 (#1).
- **Think-Then-Embed (TTE)** (2025) ★ — `arXiv:2510.05014` · [paper](https://arxiv.org/abs/2510.05014) — Reasoner→embedder; SOTA MMEB-V2; +7%.
- **RGE** (2025) — `arXiv:2511.16150` · [paper](https://arxiv.org/abs/2511.16150) — Generative rationale + contrastive; +4.9% MMEB.
- **VIRTUE** (2025) — `arXiv:2510.00523` · [paper](https://arxiv.org/abs/2510.00523) — Point/box-conditioned visual-interactive embedder on VLM.
- **ABC** (2025) — `arXiv:2503.00329` · [paper](https://arxiv.org/abs/2503.00329) — Inference-time prompt-steered VLLM embedding.
- **ModernVBERT** (2025) — `arXiv:2510.01149` · [paper](https://arxiv.org/abs/2510.01149) — Small ≤1B VLM doc retriever.
- **Llama-NemoRetriever-ColEmbed** (NVIDIA, 2025) — `arXiv:2507.05513` · [paper](https://arxiv.org/abs/2507.05513) — Top text-image late-interaction.
- **Nemotron ColEmbed V2** (NVIDIA, 2026) — `arXiv:2602.03992` · [paper](https://arxiv.org/abs/2602.03992) — SigLIP2 + Qwen3-VL late-interaction; #1 ViDoRe v3.
- **EVENT-Retriever** (2025) — `arXiv:2509.00751` · [paper](https://arxiv.org/abs/2509.00751) — Event-aware multimodal retrieval.
- **Embed-RL** (2026) — `arXiv:2602.13823` · [paper](https://arxiv.org/abs/2602.13823) — RL with reasoning rewards for multimodal embedders.

### Document / visual retrieval (ColPali family)

- **ColPali** (ICLR, 2025) ★ — `arXiv:2407.01449` · [paper](https://arxiv.org/abs/2407.01449) · [code](https://github.com/illuin-tech/colpali) — PaliGemma-3B; +29 nDCG on ViDoRe.
- **ColQwen2 / ColQwen2.5** (2024–25) — [HF](https://huggingface.co/vidore/colqwen2-v0.1) · [HF](https://huggingface.co/vidore/colqwen2.5-v0.2) — Qwen2-VL / Qwen2.5-VL multi-vector.
- **ColSmol 256M / 500M** — [HF](https://huggingface.co/vidore) — On-device.
- **Nomic Embed Multimodal 3B / 7B** (Nomic, 2025) ★ — [blog](https://www.nomic.ai/news/nomic-embed-multimodal).
- **ColNomic Embed Multimodal 3B / 7B** (Nomic, 2025) ★ — [blog](https://www.nomic.ai/news/nomic-embed-multimodal) — ViDoRe-v2 62.7.
- **ViDoRe benchmark v1 / v2** — `arXiv:2407.01449` · `arXiv:2505.17166`.

### Reasoning-intensive retrieval

- **BRIGHT** (2024) — `arXiv:2407.12883` · [site](https://brightbenchmark.github.io/) — 1,384 reasoning queries.
- **ReasonIR-8B** (2025) — `arXiv:2504.20595` · [paper](https://arxiv.org/abs/2504.20595) — 29.9 BRIGHT.
- **O1-Embedder** (2025) — `arXiv:2502.07555` · [paper](https://arxiv.org/abs/2502.07555) — Thoughts before retrieval.
- **Search-R3** (2025) — `arXiv:2510.07048` · [paper](https://arxiv.org/abs/2510.07048) — SFT + RL.
- **ReasonEmbed** (2025) — `arXiv:2510.08252` · [paper](https://arxiv.org/abs/2510.08252) — 38.1 BRIGHT (Qwen3-8B).
- **Large Reasoning Embedding Models** (2025) — `arXiv:2510.14321`.
- **RITE** (2025) — `arXiv:2509.00276`.
- **RAR-b** (2024) — `arXiv:2404.06347` · [code](https://github.com/gowitheflow-1998/RAR-b) — Reasoning-as-Retrieval benchmark.
- **RaDeR** (2025) — `arXiv:2505.18405` · [paper](https://arxiv.org/abs/2505.18405) — CoT-trajectory + self-reflective HN mining.
- **RETRO\*** (2025) — `arXiv:2509.24869` · [paper](https://arxiv.org/abs/2509.24869) — LLM optimisation for reasoning-intensive doc retrieval.
- **Search-R1** (2025) — `arXiv:2503.09516` · [paper](https://arxiv.org/abs/2503.09516) — RL-trained LLM interleaving search + reasoning.
- **R1-Searcher** (2025) — `arXiv:2503.05592` · [paper](https://arxiv.org/abs/2503.05592) — RL search agent.
- **"Your Dense Retriever is Secretly an Expeditious Reasoner"** (2025) — `arXiv:2510.21727` · [paper](https://arxiv.org/abs/2510.21727).
- **"Theoretical Limitations of Embedding-Based Retrieval"** (2025) — `arXiv:2508.21038` · [paper](https://arxiv.org/abs/2508.21038) — Formal proof on fixed-dim representational gap.

### Late-interaction / multi-vector

- **ColBERT** (SIGIR, 2020) — [paper](https://arxiv.org/abs/2004.12832).
- **ColBERTv2** (NAACL, 2022) — `arXiv:2112.01488` · [paper](https://arxiv.org/abs/2112.01488) — Residual compression.
- **PLAID** (CIKM, 2022) — `arXiv:2205.09707` · [paper](https://arxiv.org/abs/2205.09707) — Centroid pruning.
- **XTR** (NeurIPS, 2023) — `arXiv:2304.01982` · [paper](https://arxiv.org/abs/2304.01982).
- **WARP** (2025) — `arXiv:2501.17788`.
- **Jina-ColBERT-v2** (2024) — `arXiv:2408.16672`.
- **MUVERA** (NeurIPS 2024) ★ — Multi-vector → fixed-dim with provable Chamfer ε-approx; major speedup.
- **MetaEmbed** (2025) — `arXiv:2509.18095` · [paper](https://arxiv.org/abs/2509.18095) — Test-time flexible meta-tokens for late interaction.
- **jina-reranker-v3** (2025) — `arXiv:2509.25085` · [paper](https://arxiv.org/abs/2509.25085) — "Last but not late" interaction inside transformer.

---

## Omni-modal / Any-to-Any

### Binding approaches

- **ImageBind** (Meta, CVPR 2023) ★ — `arXiv:2305.05665` · [paper](https://arxiv.org/abs/2305.05665) · [code](https://github.com/facebookresearch/ImageBind) — 6 modalities, image-anchored.
- **LanguageBind** (PKU, ICLR 2024) — `arXiv:2310.01852` · [paper](https://arxiv.org/abs/2310.01852) · [code](https://github.com/PKU-YuanGroup/LanguageBind) — Language-anchored, 5 mod.
- **Meta-Transformer** (CUHK, 2023) — `arXiv:2307.10802` · [paper](https://arxiv.org/abs/2307.10802) — 12 modalities, frozen shared encoder.
- **PandaGPT** (Cambridge, 2023) — `arXiv:2305.16355` · [paper](https://arxiv.org/abs/2305.16355) — ImageBind + Vicuna + LoRA.
- **UniBind** (CVPR 2024) — `arXiv:2403.12532` · [paper](https://arxiv.org/abs/2403.12532) — LLM-augmented centric alignment.
- **FreeBind** (ICML 2024) — `arXiv:2405.04883` · [paper](https://arxiv.org/abs/2405.04883) — Space Bonds.
- **OmniBind (CVPR variant)** (2024) — `arXiv:2405.16108` · [paper](https://arxiv.org/abs/2405.16108) — 3D-audio-image-language w/ unequal-scale routing.
- **OmniBind (Nie 2024)** — `arXiv:2407.11895` · [paper](https://arxiv.org/abs/2407.11895) — 30B, remap+bind specialist spaces.
- **Point-Bind** (2023) — `arXiv:2309.00615` · [paper](https://arxiv.org/abs/2309.00615).
- **TaxaBind** (2024) — `arXiv:2411.00683` · [paper](https://arxiv.org/abs/2411.00683) — 6 ecological modalities.
- **EBind** (2025) — `arXiv:2511.14229` · [paper](https://arxiv.org/abs/2511.14229) — Small-model alternative to OmniBind.
- **WAVE** (2025) — `arXiv:2509.21990` · [paper](https://arxiv.org/abs/2509.21990) — Single MLLM produces audio+image+video embeddings jointly.
- **Anchors Aweigh!** (2024) — `arXiv:2410.02086` · [paper](https://arxiv.org/abs/2410.02086) — Anchor-modality selection theory.

### Native joint & tokenized

- **ONE-PEACE** (Alibaba, 2023) ★ — `arXiv:2305.11172` · [paper](https://arxiv.org/abs/2305.11172) · [code](https://github.com/OFA-Sys/ONE-PEACE) — 4B params, vision+audio+language truly joint.
- **4M** (EPFL/Apple, NeurIPS 2023) ★ — `arXiv:2312.06647` · [paper](https://arxiv.org/abs/2312.06647) · [code](https://github.com/apple/ml-4m) — 7 modalities, tokenised masked modeling.
- **4M-21** (EPFL/Apple, 2024) ★ — `arXiv:2406.09406` · [paper](https://arxiv.org/abs/2406.09406) — 21 modalities, 3B params.

### Audio-language

- **Wav2CLIP** (2021/22) — `arXiv:2110.11499` · [paper](https://arxiv.org/abs/2110.11499).
- **AudioCLIP** (2021/22) — `arXiv:2106.13043` · [paper](https://arxiv.org/abs/2106.13043) — Tri-modal audio-image-text.
- **CLAP (MS v1)** (2022) — `arXiv:2206.04769` · [paper](https://arxiv.org/abs/2206.04769).
- **LAION-CLAP** (2022/23) ★ — `arXiv:2211.06687` · [paper](https://arxiv.org/abs/2211.06687) · [code](https://github.com/LAION-AI/CLAP) — 630K pairs, keyword-to-caption.
- **MS-CLAP v2** (2023) — `arXiv:2309.05767` · [paper](https://arxiv.org/abs/2309.05767).
- **MuLan** (Google, 2022) — `arXiv:2208.12415` · [paper](https://arxiv.org/abs/2208.12415) — 44M music tracks.
- **MuLaP** (2021/ICASSP22) — `arXiv:2112.04214` · [paper](https://arxiv.org/abs/2112.04214).
- **T-CLAP** (2024) — `arXiv:2404.17806` · [paper](https://arxiv.org/abs/2404.17806) — Temporal negative loss.
- **CLaMP / CLaMP 3** (2023/25) — `arXiv:2304.11029` · [paper](https://arxiv.org/abs/2304.11029).
- **M2D-CLAP** (2024) — `arXiv:2406.02032` · [paper](https://arxiv.org/abs/2406.02032).
- **GLAP** (2025) — `arXiv:2506.11350` · [paper](https://arxiv.org/abs/2506.11350) — 8–145 langs sound/music/speech.
- **Audio Flamingo 2** (ICML 2025) — `arXiv:2503.03983` · [paper](https://arxiv.org/abs/2503.03983) — Long-audio LALM w/ custom CLAP.
- **Audio Flamingo 3** (2025) — `arXiv:2507.08128` · [paper](https://arxiv.org/abs/2507.08128).

### Video-language

- **VideoCLIP** (2021) — `arXiv:2109.14084`.
- **Frozen-in-Time** (2021) — `arXiv:2104.00650`.
- **CLIP4Clip** (2021) — `arXiv:2104.08860`.
- **X-CLIP** (2022) — `arXiv:2207.07285`.
- **InternVideo** (2022) — `arXiv:2212.03191`.
- **InternVideo2** (2024) ★ — `arXiv:2403.15377` · [paper](https://arxiv.org/abs/2403.15377) — 6B ViT, 260M video-text.
- **VideoPrism** (Google, 2024) — `arXiv:2402.13217` · [paper](https://arxiv.org/abs/2402.13217) — SOTA 31/33 benchmarks.
- **UMT / UMT-L** (ICCV 2023 Oral) — `arXiv:2303.16058` · [paper](https://arxiv.org/abs/2303.16058).
- **InternVid / ViCLIP** (ICLR 2024) — `arXiv:2307.06942` · [paper](https://arxiv.org/abs/2307.06942).
- **Video-ColBERT** (CVPR 2025) — Contextualised late-interaction text-to-video, dual sigmoid loss.
- **Text Is MASS** (CVPR 2024) — Stochastic text embedding for one-to-many video matching.
- **Composed Video Retrieval** (CVPR 2024) — Extends CIR to video.
- **UVRB / GenUVR** (2025) — `arXiv:2510.27571` · [paper](https://arxiv.org/abs/2510.27571) — 16-task universal video retrieval bench + 1.55M synth pairs.

### Composed image retrieval (CIR)

- **CompoDiff** (2023) — `arXiv:2303.11916` · [paper](https://arxiv.org/abs/2303.11916) — Diffusion-based ZS-CIR.
- **CIReVL** (ICLR 2024) — `arXiv:2310.09291` · [paper](https://arxiv.org/abs/2310.09291) — Training-free LLM+CLIP ZS-CIR.
- **LinCIR** (CVPR 2024) — Language-only training of ZS-CIR.
- **Knowledge-Enhanced Dual-stream ZS-CIR** (CVPR 2024) — Suo et al.
- **Generative ZS-CIR (CIG)** (CVPR 2025) — Wang & Ao.
- **DistillCIR** (ICCV 2025) — Dual-stream instruction-aware distillation.
- **CoTMR** (ICCV 2025) — Chain-of-thought multi-scale reasoning, training-free.
- **CoLLM** (Amazon, 2024) — LLM-based composed image retrieval.
- **QuRe** (2025) — `arXiv:2507.12416` · [paper](https://arxiv.org/abs/2507.12416) — Query-relevant HN sampling.
- **iSEARLE** (2024) — `arXiv:2405.02951` · [paper](https://arxiv.org/abs/2405.02951) — Improved SEARLE.
- **SpLIP** (ECCV 2024) — `arXiv:2407.04207` · [paper](https://arxiv.org/abs/2407.04207) — ZS sketch-based image retrieval.

### 3D / point-cloud

- **PointCLIP** (CVPR 2022) — `arXiv:2112.02413`.
- **PointCLIP V2** (ICCV 2023) — `arXiv:2211.11682`.
- **ULIP** (CVPR 2023) — `arXiv:2212.05171`.
- **ULIP-2** (CVPR 2024) — `arXiv:2305.08275`.
- **OpenShape** (NeurIPS 2023) — `arXiv:2305.10764` · [paper](https://arxiv.org/abs/2305.10764).
- **Uni3D** (BAAI, ICLR 2024) ★ — `arXiv:2310.06773` · [paper](https://arxiv.org/abs/2310.06773) — 1B-param 3D foundation.
- **Point-Bind** (2023) — `arXiv:2309.00615`.

---

## Domain-specialised embedders

### Biomedical

- **CheXzero** (Nat Biomed Eng 2022) — [paper](https://www.nature.com/articles/s41551-022-00936-9) — Radiologist-level X-ray ZS.
- **MedCLIP** (2022) — `arXiv:2210.10163` · [paper](https://arxiv.org/abs/2210.10163).
- **BiomedCLIP** (Microsoft, 2023) — `arXiv:2303.00915` · [paper](https://arxiv.org/abs/2303.00915) — PMC-15M.
- **PMC-CLIP** (MICCAI 2023) — `arXiv:2303.07240`.
- **UniMed-CLIP** (2024) — `arXiv:2412.10372` · [paper](https://arxiv.org/abs/2412.10372) — 6 imaging modalities.
- **QwenCLIP** (2025) — `arXiv:2511.13876` · [paper](https://arxiv.org/abs/2511.13876) — Replaces CLIP text tower w/ Qwen3-Embedding-8B for long radiology reports.
- **RadCLIP** (2024) — `arXiv:2403.09948` · [paper](https://arxiv.org/abs/2403.09948).
- **RegionMed-CLIP** (2025) — `arXiv:2508.05244` · [paper](https://arxiv.org/abs/2508.05244).
- **CLEF** (Nat. Commun. 2025) — Couples ESM2 protein reps with biological-feature contrastive.

### Remote sensing / geospatial

- **RemoteCLIP** (2023) — `arXiv:2306.11029`.
- **GeoCLIP** (NeurIPS 2023) — `arXiv:2309.16020` · [paper](https://arxiv.org/abs/2309.16020).
- **SatCLIP** (Microsoft, AAAI 2025) — `arXiv:2311.17179`.
- **GRAFT** (Cornell, 2023) — `arXiv:2312.06960` · [paper](https://arxiv.org/abs/2312.06960).
- **SkyScript** (AAAI 2024) — `arXiv:2312.12856`.
- **GeoCLOSP** (2025) — `arXiv:2507.10403` · [paper](https://arxiv.org/abs/2507.10403) — SIREN location encoder bound to RS CLIP.
- **OpenScene** (CVPR 2023) — `arXiv:2211.15654` · [paper](https://arxiv.org/abs/2211.15654) — Canonical 3D-CLIP feature distillation.

### Code embedders

- **CodeRankEmbed** (Nomic, 2024) — 137M, 8192 ctx, 21M-pair CoRNStack.
- **voyage-code-3** (Voyage, 2024) — Low-dim quantised code embedder.
- **Codestral Embed** (Mistral, 2025) — Beats Voyage Code 3 / Cohere v4 / OAI-3-large.
- **Nomic Embed Code** (Nomic, 2025) ★ — 7B SOTA on CodeSearchNet.

### Multilingual

- **bge-multilingual-gemma2** (BAAI, 2024) — Gemma2-based multilingual embedder.
- **M3DR** (2025) — `arXiv:2512.03514` — Universal multilingual visual document retrieval.
- **"Multilingual Adversarial Examples"** (2025) — `arXiv:2502.08638` — Cross-lingual adversarial eval.

---

## Training techniques

### Contrastive objectives

- **InfoNCE** (CLIP, 2021) — `arXiv:2103.00020`.
- **Sigmoid loss** (SigLIP, 2023) — `arXiv:2303.15343`.
- **Alignment + uniformity** (Wang & Isola, ICML 2020) — `arXiv:2005.10242` · [paper](https://arxiv.org/abs/2005.10242).
- **Token-wise late-interaction** (FILIP) — `arXiv:2111.07783`.
- **SPARC** — `arXiv:2401.09865`.

### Hard-negative mining

- **ANCE** (ICLR 2021) — `arXiv:2007.00808` · [paper](https://arxiv.org/abs/2007.00808).
- **RocketQA** (NAACL 2021) — `arXiv:2010.08191` · [paper](https://arxiv.org/abs/2010.08191).
- **MoCo v2** (2020) — `arXiv:2003.04297`.
- **GISTEmbed** (2024) — `arXiv:2402.16829` · [paper](https://arxiv.org/abs/2402.16829).
- **NV-Retriever / PA-HNM** (2024) — `arXiv:2407.15831` · [paper](https://arxiv.org/abs/2407.15831).

### Instruction / task conditioning

- **Instructor** — `arXiv:2212.09741`.
- **TART** — `arXiv:2211.09260`.
- **Promptriever** — `arXiv:2409.11136`.
- **FollowIR** (2024) — `arXiv:2402.14334` · [paper](https://arxiv.org/abs/2402.14334) — Instruction-following IR benchmark.
- **InstructIR** — [paper](https://arxiv.org/abs/2402.14334).

### Synthetic data generation

- **E5-Mistral synthetic (500K)** — `arXiv:2401.00368`.
- **Promptagator** (2022) — `arXiv:2209.11755` · [paper](https://arxiv.org/abs/2209.11755).
- **InPars-v2** (2023) — `arXiv:2301.01820` · [paper](https://arxiv.org/pdf/2301.01820).
- **Gecko** (Google, 2024) ★ — `arXiv:2403.20327` · [paper](https://arxiv.org/abs/2403.20327).
- **MegaPairs** — `arXiv:2412.14475`.
- **mmE5 deep-thinking** — `arXiv:2502.08468`.
- **ReMixer / ReasonEmbed** — `arXiv:2510.08252`.
- **ReasonIR synthesiser** — `arXiv:2504.20595`.

### Distillation

- **AugSBERT** — Reimers et al. (cross-enc → bi-enc).
- **DIME-FM** (2023) — `arXiv:2303.18232`.
- **TinyCLIP** — `arXiv:2309.12314`.
- **MobileCLIP / MobileCLIP2** — `arXiv:2311.17049` / `arXiv:2508.20691`.
- **Jasper + Stella multi-teacher** — `arXiv:2412.19048`.

### Matryoshka & adaptive dim

- **MRL** (NeurIPS 2022) — `arXiv:2205.13147` · [paper](https://arxiv.org/abs/2205.13147) · [code](https://github.com/RAIVNLab/MRL).
- **2D MRL / 2DMSE** — `arXiv:2402.14776`.
- **Starbucks / v2** — `arXiv:2410.13230`.
- **2D MRL for IR** — `arXiv:2411.17299`.
- **CSR — Beyond Matryoshka** — `arXiv:2503.01776`.
- **SMEC** — `arXiv:2510.12474`.
- **Quantization for RAG (4-bit)** — `arXiv:2501.10534`.
- **Embedding Quantization HF blog** — [blog](https://huggingface.co/blog/embedding-quantization).

### Pooling & causal→bidirectional conversion

- **LLM2Vec (bidir + MNTP + SimCSE)** — `arXiv:2404.05961`.
- **Echo (input repetition)** — `arXiv:2402.15449`.
- **Causal2Vec (prepended contextual token)** — `arXiv:2507.23386`.
- **NV-Embed latent-attention pooling** — `arXiv:2405.17428`.
- **Pooling-and-Attention study** — `arXiv:2409.02727`.

### PEFT / LoRA

- **jina-v3 five task-LoRAs** — `arXiv:2409.10173`.
- **jina-v4 three task-LoRAs (MV + dense)** — `arXiv:2506.18902`.

---

## Benchmarks

### Text

- **MTEB** (EACL 2023) — `arXiv:2210.07316` · [leaderboard](https://huggingface.co/spaces/mteb/leaderboard).
- **MMTEB** (ICLR 2025) ★ — `arXiv:2502.13595` — 500+ tasks, 250+ langs.
- **MTEB-Eng v2** — subset of MMTEB.
- **C-MTEB** — `arXiv:2309.07597`.
- **MTEB-French** — `arXiv:2405.20468`.
- **MTEB Arena** — [HF space](https://huggingface.co/spaces/mteb/arena).
- **BEIR** (NeurIPS 2021 D&B) — `arXiv:2104.08663`.
- **BRIGHT** — `arXiv:2407.12883` · [site](https://brightbenchmark.github.io/).
- **LoCo** — `arXiv:2402.07440`.
- **LongEmbed** — `arXiv:2404.12096`.
- **RAR-b** — `arXiv:2404.06347`.
- **MAIR** — `arXiv:2410.10127`.
- **RTEB** (HF, 2025) ★ — [blog](https://huggingface.co/blog/rteb) · private held-out sets.
- **CoIR** (code) — `arXiv:2407.02883`.
- **FollowIR** — `arXiv:2403.15246`.

### Multimodal

- **MMEB / MMEB-V2** ★ — `arXiv:2410.05160` / `arXiv:2507.04590` · [leaderboard](https://huggingface.co/spaces/TIGER-Lab/MMEB-Leaderboard).
- **MMDocIR** (2025) — `arXiv:2501.08828` · [paper](https://arxiv.org/abs/2501.08828) — Long multimodal document retrieval.
- **TARGET** (2025) — `arXiv:2505.11545` · [paper](https://arxiv.org/abs/2505.11545) — Table retrieval for generative tasks.
- **SEA-BED** (2025) — `arXiv:2508.12243` · [paper](https://arxiv.org/abs/2508.12243) — Southeast-Asian language embedding eval.
- **NegBench** (CVPR 2025) — `arXiv:2501.09425` — Negation-stress benchmark.
- **VisMin** (NeurIPS 2024) — Minimal-change visual benchmark.
- **UVRB** (2025) — `arXiv:2510.27571` — Universal video retrieval (16 tasks).
- **LoVR** (2025) — `arXiv:2505.13928` · [paper](https://arxiv.org/abs/2505.13928) — Long-video retrieval bench.
- **CARE-Bench** (2025) — `arXiv:2501.00513` · [paper](https://arxiv.org/abs/2501.00513) — Fine-grained video captioning + retrieval.
- **M-BEIR** — `arXiv:2311.17136` · [project](https://tiger-ai-lab.github.io/UniIR/).
- **MIEB** — `arXiv:2504.10471` — 130 tasks, 38 langs.
- **ViDoRe v1 / v2** ★ — `arXiv:2407.01449` / `arXiv:2505.17166`.
- **UMRB** — `arXiv:2412.16855`.
- **Flickr30K** / **MS-COCO** (classical).
- **M2KR** — `arXiv:2402.08327`.
- **CIRR** / **CIRCO** / **FashionIQ** — composed IR.
- **InfoSeek** / **OVEN** / **Encyclopedic-VQA** — knowledge-intensive VQA-retrieval.
- **OmniBench** — `arXiv:2409.15272` — tri-modal reasoning.

### Video / audio / 3D

- **MSR-VTT, MSVD, LSMDC, VATEX, ActivityNet-Captions, DiDeMo, YouCook2** — classical video retrieval.
- **MVBench** (CVPR 2024) — `arXiv:2311.17005`.
- **Video-MME** (CVPR 2025) — `arXiv:2405.21075`.
- **AudioSet** (ICASSP 2017).
- **VGGSound** (ICASSP 2020).
- **AudioCaps** (NAACL 2019).
- **Clotho v2** (ICASSP 2020).
- **ESC-50** (2015) · **UrbanSound8K** · **FSD50K**.
- **AudioBench** — `arXiv:2406.16020`.
- **Objaverse-LVIS, ModelNet40, ScanObjectNN** — 3D ZS.

### Compositionality & robustness

- **Winoground** (CVPR 2022).
- **SugarCrepe / ++** — `arXiv:2306.14610` / `arXiv:2406.11171`.
- **CREPE** / **ARO** (hackable) — `arXiv:2212.07796` / `arXiv:2210.01936`.
- **EqBen** (ICCV 2023 oral).
- **VALSE** (ACL 2022).
- **NaturalBench, MMVP, BiVLC** — fine-grained probes.
- **ELEVATER** (NeurIPS 2022 D&B) — `arXiv:2204.08790`.

---

## Datasets (training)

| Dataset | Scale | Year | Notes |
|---|---|---|---|
| CC3M | 3.3M | 2018 | Conceptual Captions |
| CC12M | 12M | 2021 | `arXiv:2102.08981` |
| YFCC100M | 100M | 2015 | `arXiv:1503.01817` |
| WIT (Wikipedia) | 37.6M / 108 langs | 2021 | `arXiv:2103.01913` |
| WIT-400M (OpenAI) | 400M | 2021 | internal |
| ALIGN-1.8B | 1.8B | 2021 | internal |
| LAION-400M | 400M | 2021 | `arXiv:2111.02114` |
| LAION-2B-en | 2.32B | 2022 | `arXiv:2210.08402` |
| LAION-5B | 5.85B | 2022 | same |
| COYO-700M | 747M | 2022 | Kakao Brain |
| DataComp-1B | 1.28B | 2023 | `arXiv:2304.14108` |
| DFN-2B / 5B | 2B / 5B | 2023 | `arXiv:2309.17425` |
| Recap-DataComp-1B | 1.3B re-caps | 2024 | `arXiv:2406.08478` |
| VeCap-300M | 300M | 2023 | `arXiv:2310.07699` |
| WebLI | ~10B / 109 langs | 2022 | `arXiv:2209.06794` |
| MetaCLIP-2.5B / 5.4B | up to 5.4B | 2023–25 | `arXiv:2309.16671` / `arXiv:2507.22062` |
| ShareGPT4V | 1.2M | 2023 | `arXiv:2311.12793` |
| DreamLIP-30M | 30M re-caps | 2024 | `arXiv:2403.17007` |
| DCI | 7,805 × 1K-word | 2024 | `arXiv:2312.08578` |
| VLM-150M (HQ-CLIP) | 150M | 2025 | `arXiv:2507.22431` |
| WebVid-10M | 10.7M clips | 2021 | `arXiv:2104.00650` |
| HowTo100M | 136M clips | 2019 | `arXiv:1906.03327` |
| InternVid | 234M clips | 2023 | `arXiv:2307.06942` |
| FLD-5B (Florence-2) | 5.4B annot / 126M imgs | 2023 | `arXiv:2311.06242` |
| MMEB-train | ~700K | 2024 | VLM2Vec |
| E5 synthetic | 500K / 93 langs | 2024 | `arXiv:2401.00368` |
| MegaPairs | 26M triplets | 2024 | `arXiv:2412.14475` |
| mmE5 synth | 560K | 2025 | `arXiv:2502.08468` |
| AudioSet | 2M clips | 2017 | — |
| WavCaps | ~400K audio-text | 2023 | — |
| LAION-Audio-630K | 630K | 2022 | LAION-CLAP |
| Objaverse-LVIS | ~50K | 2023 | OpenShape |
| BioMed / PMC-15M | 15M img-text | 2023 | BiomedCLIP |
| UniMed-5.3M | 5.3M / 6 mod | 2024 | `arXiv:2412.10372` |

---

## Surveys & positional papers

- **Liang et al. "Mind the Gap"** (NeurIPS 2022) — `arXiv:2203.02053` — Modality gap analysis.
- **Shi et al. "It's Not a Modality Gap"** (2024) — `arXiv:2405.18570` — Contrastive-gap reframing.
- **"Maintaining MTEB"** (2025) — `arXiv:2506.21182` — Contamination analysis.
- **VLM survey** (Zong et al., 2023) — general VLM overview.
- **Multimodal LLM survey** (Yin et al., 2024) — covers MLLMs but not embedding-specific.
- **This work** — [survey.md](survey.md).

---

## ICLR 2026 / NeurIPS 2025 frontier

### Reasoning retrieval & test-time scaling

- **Q-RAG** (ICLR 2026 Outstanding) ★ — `arXiv:2511.07328` · [paper](https://arxiv.org/abs/2511.07328) — Value-based RL retriever; 10M-token BabiLong/RULER SOTA.
- **MetaEmbed** (ICLR 2026) — `arXiv:2509.18095` · [paper](https://arxiv.org/abs/2509.18095) — Test-time-flexible Meta-Tokens + Matryoshka multi-vector.
- **UME-R1** (ICLR 2026) — `arXiv:2511.00405` · [paper](https://arxiv.org/abs/2511.00405) — Two-stage SFT+RL discriminative + generative embeddings.
- **HRSA "Do Reasoning Models Enhance Embeddings?"** (2026) ★ — `arXiv:2601.21192` · [paper](https://arxiv.org/abs/2601.21192) — **Contrarian foil**: RLVR-init → identical embeddings post-contrastive.

### KG-RAG cluster

- **GFM-RAG** (NeurIPS 2025) ★ — `arXiv:2502.01113` · [paper](https://arxiv.org/abs/2502.01113) — 8M-param query-dependent GNN over 60 KGs / 14M triples.
- **HyperGraphRAG** (NeurIPS 2025) — `arXiv:2503.21322` · [paper](https://arxiv.org/abs/2503.21322) — N-ary hyperedge relations.
- **GraphRAG-Bench** (ICLR 2026) — `arXiv:2506.05690` · [paper](https://arxiv.org/abs/2506.05690) — When graphs help RAG (with negatives).
- **LinearRAG** (ICLR 2026) — `arXiv:2510.10114` · [paper](https://arxiv.org/abs/2510.10114).
- **DEG-RAG** (ICLR 2026) — `arXiv:2510.14271` · [paper](https://arxiv.org/abs/2510.14271).
- **UltRAG** (ICLR 2026) — `arXiv:2603.28773` · [paper](https://arxiv.org/abs/2603.28773) — Wikidata-scale 116M entities.
- **T²RAG** (2025) — `arXiv:2508.02435` · [paper](https://arxiv.org/abs/2508.02435).
- **PropRAG** (EMNLP 2025) — `arXiv:2504.18070` · [paper](https://arxiv.org/abs/2504.18070) — Beam search over proposition paths.

### Agentic RAG

- **AceSearcher** (NeurIPS 2025 Spotlight) — `arXiv:2509.24193` · [paper](https://arxiv.org/abs/2509.24193) — Decomposer/solver self-play.
- **ReasonRAG (RAG-ProGUIDE)** (NeurIPS 2025) — `arXiv:2505.14069` · [paper](https://arxiv.org/abs/2505.14069) — Process-vs-outcome RL reward.
- **MMOA-RAG** (NeurIPS 2025) — `arXiv:2501.15228` · [paper](https://arxiv.org/abs/2501.15228) — Multi-agent RL.
- **CoopRAG** (NeurIPS 2025) — `arXiv:2512.10422` · [paper](https://arxiv.org/abs/2512.10422).
- **Wiki-PRF** (NeurIPS 2025) — `arXiv:2510.14605` · [paper](https://arxiv.org/abs/2510.14605) — RL on multimodal RAG; SOTA E-VQA + InfoSeek.
- **Vgent** (NeurIPS 2025 Spotlight) — `arXiv:2510.14032` · [paper](https://arxiv.org/abs/2510.14032) — Graph RAG for long video.
- **End-to-End Multimodal RAG via Reward Backprop** (EMNLP 2025) — [aclanthology .24](https://aclanthology.org/2025.findings-emnlp.24/).

### Diffusion-as-retriever

- **DiffuGR** (2025) — `arXiv:2511.08150` · [paper](https://arxiv.org/abs/2511.08150) — Discrete diffusion DocID generation.
- **DiffGRM** (2025) — `arXiv:2510.21805` · [paper](https://arxiv.org/abs/2510.21805) — Diffusion semantic IDs for recsys.

### Composed retrieval & unified MLLM embedders 2025-26

- **PinPoint** (CVPR 2026) — `arXiv:2603.04598` · [paper](https://arxiv.org/abs/2603.04598) — 7635-query composed retrieval w/ paraphrase variance.
- **M3T-UEM** (ICCV 2025) — Task-adaptive M-BEIR.
- **UniMoCo** (2025) — `arXiv:2505.11815` — Modality-completion robustness.
- **Bridging Modalities (GME paper-form)** (CVPR 2025) — synthetic fused-modal data + UMRB.

### EMNLP 2025 — text embedders & rerankers

- **Conan-Embedding-v2** (Tencent, EMNLP 2025 Main) ★ — `arXiv:2509.12892` · [paper](https://arxiv.org/abs/2509.12892) — 1.4B from-scratch LLM embedder; SOTA MTEB + C-MTEB.
- **REARANK** (EMNLP 2025) — `arXiv:2505.20046` · [paper](https://arxiv.org/abs/2505.20046) — 7B listwise reranker matches GPT-4 with 179 RL samples.
- **Static Word Embeddings for Sentence Reps** (EMNLP 2025) — `arXiv:2506.04624`.
- **Co-Evolving LLMs + Embedder via Density-Guided DPO** (EMNLP 2025) — [aclanthology .241](https://aclanthology.org/2025.emnlp-main.241/).
- **Cross-Lingual LLM Retrieval Eval** (EMNLP 2025) — `arXiv:2509.14749`.
- **SparseCL** (ICML 2025) — Contradiction retrieval via contrastive sparsity.

### Benchmarks

- **MMDocRAG** (NeurIPS 2025) — `arXiv:2505.16470` · [paper](https://arxiv.org/abs/2505.16470) — 4055 QAs × multi-page evidence × 60 VLMs × 14 retrievers.
- **FinMTEB** (EMNLP 2025) — `arXiv:2502.10990` — Finance text embedding (en/zh).
- **MathNet** (ICLR 2026) — `arXiv:2604.18584` — Olympiad, 47 countries × 17 langs.
- **MultiTableQA via T-RAG** (ICLR 2026) — `arXiv:2504.01346` — 57K tables × 23K Qs.

### Frontier closed/open models 2025-26

- **Cohere Rerank 4** (Apr 2025) — [blog](https://cohere.com/blog/rerank-4) — 128K ctx, MRL [256–1536].
- **voyage-multimodal-3.5 / Voyage 4 (MoE)** (Jan 2026) — [blog](https://blog.voyageai.com/2026/01/15/voyage-multimodal-3-5/) — +4.56% over Cohere v4 on visual doc.
- **Granite Embedding R2** (IBM 2025) — `arXiv:2508.21085` — ModernBERT 47M/149M Apache-2.0.
- **mxbai-rerank v2** (Mixedbread 2025) — [blog](https://www.mixedbread.com/blog/mxbai-rerank-v2) — GRPO-trained 0.5/1.5B.
- **BGE-VL-MLLM-S1/S2** (BAAI Mar 2025) — [HF](https://huggingface.co/BAAI/BGE-VL-v1.5-mmeb).

### Theory

- **ECI (Effective Contrastive Information)** (2026) — `arXiv:2603.20990` — Information-theoretic HN-quality metric.
- **Temperature-Free InfoNCE** (2025) — `arXiv:2501.17683`.
- **Hard Negatives, Hard Lessons** (EMNLP 2025) — [aclanthology .481](https://aclanthology.org/2025.findings-emnlp.481.pdf) — 58% of pairs have ≥1 false negative.

### Index / GPU infrastructure

- **CAGRA + cuVS GPU graph indexing** (NVIDIA 2025-26) — `arXiv:2508.08744` — 12× faster build, CAGRA→HNSW interop.

---

## Universal audio SSL & speech foundations

- **BEATs** (ICML 2023) — `arXiv:2212.09058` · [paper](https://arxiv.org/abs/2212.09058) — Iterative acoustic tokeniser + masked audio modeling.
- **M2D** (ICASSP 2023) — `arXiv:2210.14648` · [paper](https://arxiv.org/abs/2210.14648).
- **ATST** (ICASSP 2023) — `arXiv:2204.12076` · [paper](https://arxiv.org/abs/2204.12076).
- **Dasheng** (Interspeech 2024) ★ — `arXiv:2406.06992` · [paper](https://arxiv.org/abs/2406.06992) — 1.2B-param SSL on 272K hours; SOTA HEAR.
- **AudioMAE** (NeurIPS 2022) — `arXiv:2207.06405` · [paper](https://arxiv.org/abs/2207.06405).
- **OWSM v3.1 / v3.2** (Interspeech 24 / ICASSP 25) — `arXiv:2406.09282` / `arXiv:2502.10373`.
- **USM** (Google 2023) — `arXiv:2303.01037` — 2B-param 100+ langs.
- **WavLM-Large** (JSTSP 2022) — `arXiv:2110.13900`.

## Speaker embedders

- **WeSpeaker** (ICASSP 2023) — `arXiv:2210.17016` — Open speaker-embedding toolkit.
- **CAM++ / ERes2Net** (Interspeech 2023) — `arXiv:2303.00332` / `arXiv:2305.12838`.
- **RedimNet** (Interspeech 2024) — `arXiv:2407.18223`.

## Speech-LLM unified (audio-language)

- **SALMONN** (ICLR 2024) ★ — `arXiv:2310.13289` · [paper](https://arxiv.org/abs/2310.13289) — Whisper + BEATs dual-encoder unified audio-LLM.
- **AudioPaLM** (Google 2023) — `arXiv:2306.12925`.
- **Pengi** (NeurIPS 2023) — `arXiv:2305.11834`.
- **Cacophony** (ICASSP 2024) — `arXiv:2402.06986` — Improved CLAP recipe.
- **DCASE 2024 Task 8 (Language-Based Audio Retrieval)** — [DCASE](https://dcase.community/challenge2024/task-language-based-audio-retrieval).

## Video MLLM 2024–25

- **VideoMAE V2** (CVPR 2023) — `arXiv:2303.16727`.
- **VideoChat2 + MVBench** (CVPR 2024) — `arXiv:2311.17005`.
- **LongVU** (Meta 2024) — `arXiv:2410.17434`.
- **Tarsier / Tarsier2** (ByteDance 24-25) ★ — `arXiv:2407.00634` / `arXiv:2501.07888`.
- **Video-CCAM** (Tencent 2024) — `arXiv:2408.14023`.
- **PLLaVA** (2024) — `arXiv:2404.16994`.
- **MovieChat** (CVPR 2024) — `arXiv:2307.16449` — 10K-frame long video.

## Cross-modal contrastive theory

- **Spectral Contrastive Loss** (NeurIPS 2021) — `arXiv:2106.04156`.
- **"Inductive Biases" (Saunshi)** (ICML 2022) — `arXiv:2202.14037`.
- **MERU** (ICML 2023) — `arXiv:2304.09172` — Hyperbolic CLIP.
- **HypCD** (CVPR 2024) — `arXiv:2404.14240`.
- **DCL** (ECCV 2022) — `arXiv:2110.06848`.
- **PCL** (ICLR 2021) — `arXiv:2005.04966`.
- **Scaling Laws for Dense Retrieval** (SIGIR 2024) — `arXiv:2403.18684`.

## Embodied / robotics / tactile / event

- **VIP** (ICLR 2023) — `arXiv:2210.00030`.
- **R3M** (CoRL 2022) — `arXiv:2203.12601`.
- **Voltron** (RSS 2023) — `arXiv:2302.12766`.
- **RoboCLIP** (NeurIPS 2023) — `arXiv:2310.07899`.
- **OpenVLA** (CoRL 2024) ★ — `arXiv:2406.09246`.
- **RT-2** (CoRL 2023) — `arXiv:2307.15818`.
- **Octo** (RSS 2024) — `arXiv:2405.12213`.
- **TVL (Touch-Vision-Language)** (ICML 2024) ★ — `arXiv:2402.13232`.
- **Sparsh** (CoRL 2024) — `arXiv:2410.24090`.
- **E-CLIP** (2023) — `arXiv:2310.02495`.
- **EventCLIP** (2023) — `arXiv:2306.06354`.

## Time-series foundation embedders

- **MOMENT** (ICML 2024) — `arXiv:2402.03885`.
- **Chronos** (TMLR 2024) — `arXiv:2403.07815`.
- **MOIRAI** (ICML 2024) — `arXiv:2402.02592`.
- **TimesFM** (ICML 2024) — `arXiv:2310.10688`.

## Bio: protein / molecule

- **ESM-3** (Science 2025) ★ — bioRxiv 2024.07.01.600583 — Multimodal protein embedder.
- **Boltz-1 / Boltz-2** (MIT 2024-25) — bioRxiv 2024.11.19.624167.
- **MoLFormer-XL** (Nat. Mach. Intel. 2022) — `arXiv:2106.09553`.

## Legal / scientific / finance / DocAI

- **SaulLM-7B/54B/141B** (Equall 2024) — `arXiv:2403.03883` / `arXiv:2407.19584`.
- **SPECTER2** (ACL 2024) — `arXiv:2211.13308`.
- **SciNCL** (EMNLP 2022) — `arXiv:2202.06671`.
- **OAG-BERT** (KDD 2022) — `arXiv:2103.02410`.
- **FinMA / FinGPT** (NeurIPS 2023) — `arXiv:2306.05443`.
- **LayoutLMv3** (ACM MM 2022) — `arXiv:2204.08387`.
- **DocLLM** (ACL 2024) — `arXiv:2401.00908`.

## Pathology / CT (medical 2024)

- **RadFM** (2023) — `arXiv:2308.02463`.
- **Virchow 2** (Paige AI 2024) — `arXiv:2408.00738`.
- **UNI** (Nature Med. 2024) ★ — [paper](https://www.nature.com/articles/s41591-024-02857-3).
- **CONCH** (Nature Med. 2024) — [paper](https://www.nature.com/articles/s41591-024-02856-4).
- **CT-CLIP** (2024) — `arXiv:2403.17834`.

## Remote sensing / geospatial 2024

- **SatMAE++** (CVPR 2024) — `arXiv:2403.05419`.
- **Prithvi / Prithvi-EO-2.0** (IBM/NASA) — `arXiv:2310.18660` / `arXiv:2412.02732`.

## Sentence-embedding mining

- **ESimCSE** (COLING 2022) — `arXiv:2109.04380`.
- **DiffCSE** (NAACL 2022) — `arXiv:2204.10298`.
- **PromptBERT** (EMNLP 2022) — `arXiv:2201.04337`.

## RAG / long-doc benchmarks

- **CRAG** (NeurIPS 2024 D&B) — `arXiv:2406.04744`.
- **CRUD-RAG** (TOIS 2024) — `arXiv:2401.17043` — Chinese RAG.
- **RGB** (AAAI 2024) — `arXiv:2309.01431`.
- **MMLongBench-Doc** (NeurIPS 2024) — `arXiv:2407.01523`.
- **ViDoSeek** (2025) — `arXiv:2505.17496`.

## Generative retrieval (parallel track to dense embedding)

- **GenRet** (NeurIPS 2023) — `arXiv:2304.04171` · [paper](https://arxiv.org/abs/2304.04171) — Joint tokenizer + retrieval objective for discrete docids.
- **LTRGR** (AAAI 2024) — `arXiv:2306.15222` · [paper](https://arxiv.org/abs/2306.15222) — Passage-rank loss on AR docid generation.
- **RIPOR** (WWW 2024) — `arXiv:2311.09134` · [paper](https://arxiv.org/abs/2311.09134) — Residual quantisation + prefix-oriented relevance tuning.
- **GLEN** (EMNLP 2023) — `arXiv:2311.03057` · [paper](https://arxiv.org/abs/2311.03057) — Lexical (keyword) docids.
- **NOVO** (CIKM 2023) — N-gram learnable docids.
- **ASI** (EMNLP 2023) — `arXiv:2310.11015` · [paper](https://arxiv.org/abs/2310.11015) — End-to-end semantic docids w/o offline clustering.
- **MINDER** (ACL 2023) — `arXiv:2305.16675` · [paper](https://arxiv.org/abs/2305.16675) — Multiview docids.
- **GenRRL** (SIGIR 2024) — `arXiv:2407.05316` · [paper](https://arxiv.org/abs/2407.05316) — RL with relevance feedback for generative IR.
- **DiffusionRet** (SIGIR 2024 short) — `arXiv:2305.09515` · [paper](https://arxiv.org/abs/2305.09515) — Diffusion process for retrieval.
- **TIGER** (NeurIPS 2023) ★ — `arXiv:2305.05065` · [paper](https://arxiv.org/abs/2305.05065) — Generative recommender w/ RQ-VAE semantic IDs (also under recsys).

## Cross-encoder rerankers (deployment companion)

- **RankZephyr** (2023) — `arXiv:2312.02724` · [paper](https://arxiv.org/abs/2312.02724) — Distill GPT-4 listwise → Zephyr-7B.
- **RankLLaMA** (SIGIR 2024) — `arXiv:2310.08319` · [paper](https://arxiv.org/abs/2310.08319) — Pointwise LLaMA reranker.
- **RankVicuna** (2023) — `arXiv:2309.15088` · [paper](https://arxiv.org/abs/2309.15088).
- **Setwise Prompting** (SIGIR 2024) — `arXiv:2310.09497` · [paper](https://arxiv.org/abs/2310.09497) — Tournament-style prompting.
- **PRP (Pairwise Ranking Prompting)** (NAACL 2024 Findings) — `arXiv:2306.17563` · [paper](https://arxiv.org/abs/2306.17563).
- **bge-reranker-v2-gemma / v2-m3 / v2.5-gemma2-lightweight** (BAAI HF 2024) ★ — [HF](https://huggingface.co/BAAI/bge-reranker-v2-gemma) — Multilingual cross-encoder family.
- **Qwen3-Reranker 0.6B / 4B / 8B** (Alibaba 2025) ★ — `arXiv:2506.05176` · [HF](https://huggingface.co/Qwen/Qwen3-Reranker-8B) — Reranker family paired with Qwen3-Embedding.
- **jina-reranker-v2-base-multilingual** (Jina 2024) — [HF](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual).
- **jina-reranker-m0 (multimodal)** (Jina 2025) ★ — [HF](https://huggingface.co/jinaai/jina-reranker-m0) — First open multimodal cross-encoder reranker.
- **Cohere Rerank 3 / 3.5** — [blog](https://cohere.com/blog/rerank-3pt5).
- **Voyage Rerank-2 / Rerank-2-Lite** — [blog](https://blog.voyageai.com/2024/09/30/rerank-2/) — 16K context RAG-tuned.
- **FIRST** (EMNLP 2024) — `arXiv:2406.15657` · [paper](https://arxiv.org/abs/2406.15657) — Single-token logit listwise reranking.
- **FLARE reranker** (SIGIR 2024) — `arXiv:2308.12086` · [paper](https://arxiv.org/abs/2308.12086).
- **TourRank** (SIGIR 2025) — `arXiv:2406.11678` · [paper](https://arxiv.org/abs/2406.11678).
- **LiT5-Distill / LiT5-Score** (SIGIR 2024) — `arXiv:2312.16098` · [paper](https://arxiv.org/abs/2312.16098).

## Privacy / robustness / fairness

- **vec2text** (EMNLP 2023) ★ — `arXiv:2310.06816` · [paper](https://arxiv.org/abs/2310.06816) — Iterative decoder inverts embeddings to text.
- **GEIA** (ACL 2023 Findings) — `arXiv:2305.03010` · [paper](https://arxiv.org/abs/2305.03010) — Generative embedding inversion.
- **Information Leakage from LLM Embeddings** (2024) — `arXiv:2405.11916`.
- **DP-SGD for Dense Retrieval** (SIGIR 2024) — `arXiv:2402.07023` · [paper](https://arxiv.org/abs/2402.07023).
- **Defending Against Embedding Inversion via Noise Injection** (EMNLP 2024) — `arXiv:2404.12014`.
- **Multilingual Text Embedding Inversion Security** (ACL 2024) — `arXiv:2401.12192`.
- **PRADA — Adversarial Attacks on Dense Retrieval** (SIGIR 2023) — `arXiv:2204.01321`.
- **FAIR-IR — Fairness in Dense Retrieval** (SIGIR 2024) — `arXiv:2404.17357`.

## Recommendation-as-embedding

- **TIGER** (NeurIPS 2023) — `arXiv:2305.05065`.
- **COBRA** (KDD 2024 / WWW 2025) — `arXiv:2503.02453` · [paper](https://arxiv.org/abs/2503.02453) — Hierarchical semantic IDs.
- **LC-Rec** (ICDE 2024) — `arXiv:2311.09049`.
- **LLaRA** (SIGIR 2024) — `arXiv:2312.02445`.
- **E4SRec** (WWW 2024) — `arXiv:2312.02443`.
- **HSTU** (Meta, ICML 2024) ★ — `arXiv:2402.17152` · [paper](https://arxiv.org/abs/2402.17152) — Trillion-param sequential transducers for generative recsys.
- **CoLLM (recsys)** (TKDE 2025) — `arXiv:2310.19488`.

## RAG-recipe layer

- **Self-RAG** (ICLR 2024) ★ — `arXiv:2310.11511` · [paper](https://arxiv.org/abs/2310.11511) — Reflective tokens control retrieve/generate.
- **Adaptive-RAG** (NAACL 2024) — `arXiv:2403.14403`.
- **REPLUG** (NAACL 2024) — `arXiv:2301.12652`.
- **GraphRAG** (MSR 2024) ★ — `arXiv:2404.16130` · [paper](https://arxiv.org/abs/2404.16130) — KG index + community summaries.
- **LightRAG** (2024) — `arXiv:2410.05779`.
- **HippoRAG** (NeurIPS 2024) — `arXiv:2405.14831`.
- **G-Retriever** (NeurIPS 2024) — `arXiv:2402.07630`.
- **Anthropic Contextual Retrieval** (Sep 2024) — [blog](https://www.anthropic.com/news/contextual-retrieval) — Per-chunk LLM context annotation before embedding.
- **InstructRetro** (ICLR 2024) — `arXiv:2310.07713`.
- **CONQRR** (EMNLP 2022) — `arXiv:2112.08558` — RL conversational rewriter.
- **ConvGQR** (ACL 2023) — `arXiv:2305.15645`.
- **LLM4CS** (EMNLP 2023) — `arXiv:2305.06573`.
- **RAGChecker** (NeurIPS 2024 D&B) — `arXiv:2408.08067`.

## Vision-tower foundations (used as embedders)

- **RADIO / AM-RADIO** (NVIDIA, CVPR 2024) — `arXiv:2312.06709` · [paper](https://arxiv.org/abs/2312.06709) — CLIP+DINOv2+SAM agglomerative.
- **RADIOv2.5** (2024) — `arXiv:2412.07679` · [paper](https://arxiv.org/abs/2412.07679).
- **Theia** (CoRL 2024) — `arXiv:2407.20179`.
- **InternViT-300M-V2.5** (Shanghai AI Lab) — `arXiv:2412.05271` · [paper](https://arxiv.org/abs/2412.05271).

## More foundation embedders

- **Granite Embedding** (IBM 2024) — `arXiv:2502.20204` · [paper](https://arxiv.org/abs/2502.20204) — 30M / 107M / 278M, Apache-2.0.
- **Inf-Retriever-v1** (Inf 2025) — [HF](https://huggingface.co/infly/inf-retriever-v1).
- **KaLM-Embedding** (HIT-SCIR 2025) — `arXiv:2501.01028`.
- **Conan-Embedding** (Tencent 2024) — `arXiv:2408.15710` — Dynamic HN mining.
- **gte-multilingual-base** (Alibaba 2024) — `arXiv:2407.19669`.
- **mxbai-embed-large-v1** (Mixedbread AI 2024) — [HF](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1).
- **Yi-Embedding** (01-AI 2024) — Chinese-strong bilingual.
- **GTE-ModernColBERT-v1** (LightOn 2025) — [HF](https://huggingface.co/lightonai/GTE-ModernColBERT-v1).
- **M2-BERT Retrieval** (Together+Stanford 2024) — `arXiv:2402.07440` — Monarch-Mixer sub-quadratic.
- **Marqo-FashionCLIP / FashionSigLIP** (Marqo 2024) — domain-specific fashion.

## Sparse / multi-vector @ IR venues

- **EMVB** (SIGIR 2024) — `arXiv:2404.02805` · [paper](https://arxiv.org/abs/2404.02805) — Centroid + bit-vector ColBERT pruning.
- **ConstBERT** (SIGIR 2024) — `arXiv:2405.04732`.
- **DyVo** (EMNLP 2024) — `arXiv:2410.07949` — Entity-token learned-sparse.
- **JaColBERT v1/v2/v2.5** — `arXiv:2407.20750`.
- **ColBERT-PRF** (TOIS 2023) — Token-level pseudo-relevance feedback.
- **SPLADE-v3** (Naver 2024) — `arXiv:2403.06789`.
- **RetroMAE-2** (BAAI ACL 2023) — `arXiv:2305.02564`.

## Audio / 3D extensions

- **Qwen2-Audio** (Alibaba 2024) — `arXiv:2407.10759`.
- **SpiRit-LM** (Meta TACL 2024) — `arXiv:2402.05755` — Interleaved spoken+written tokens.
- **SLAM-LLM** (2024) — `arXiv:2402.08846`.
- **GAMA** (EMNLP 2024) — `arXiv:2406.11768`.
- **PointLLM** (ECCV 2024 Oral) — `arXiv:2308.16911`.
- **OpenMask3D** (NeurIPS 2023) — `arXiv:2306.13631`.

## Theory

- **Hard-Negative Asymmetry** (TMLR 2024) — `arXiv:2305.15848`.
- **Listwise Distillation w/ Margin-MSE** (SIGIR 2024) — `arXiv:2404.11731`.

---


## License

This list is released under CC0 / Public Domain. Individual papers, code, and models retain their own licenses — please consult each source before use.
