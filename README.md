# E-VAPORISER — Contraband Image Classifier & Seller Portal

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-FFD21E?logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-F97316?logo=gradio&logoColor=white)
![Weights & Biases](https://img.shields.io/badge/Weights%20%26%20Biases-FFBE00?logo=weightsandbiases&logoColor=black)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

Automated contraband screening for online classified-ads marketplaces. This system flags
listing photos showing an **e-vaporiser (vape)** so prohibited items can be blocked before
going live. Features an empirical transfer-learning benchmark comparing **ResNet-18** vs
**ViT-B/16**, hard-negative curation against lookalike objects, and a live two-model
**Gradio seller portal** with an automated triage verdict.

> A marketplace moderation filter cannot rely on easy image classification. A naive model easily
> cheats via shortcut learning (predicting background resolution or camera noise rather than
> contraband). The real job is to tell **contraband from benign lookalikes** — pens, flashlights,
> lighters, remotes, and power banks — without introducing a manual moderation bottleneck.

## What it does
- 🔍 **Contraband detection on marketplace listings** — binary visual classifier flagging
  e-vaporisers in product photos before listings publish.
- 🛡️ **Hard-negative robustness** — trained on 800 real vape photos balanced against 800
  negatives (50% resolution-matched everyday scenes from Imagenette-320 + 50% hard negatives from
  Open Images: pens, flashlights, remotes, microphones).
- ⚖️ **Dual-backbone benchmark** — empirical comparison between a classic CNN (`ResNet-18`, 11.7M
  parameters) and a Vision Transformer (`ViT-B/16`, 86M parameters), testing both full fine-tuning
  and frozen head-only strategies across four identical training pipelines.
- 🎯 **Moderation-aware threshold tuning** — evaluates precision vs recall tradeoffs rather than
  raw accuracy alone; threshold tuned on validation data to achieve $\ge 99\%$ vape recall.
- 🛍️ **Dual-model seller portal** (Gradio) — uploads are scored live by both top models
  side-by-side with confidence bars and a three-tier verdict: **Blocked** (both agree), **Needs
  Review** (models disagree), or **Allowed** (both clear).

## Architecture
```
listing photo ──▶ [seeded resize + normalisation] ──▶ dual-model inference
                                                               │
                                  ┌────────────────────────────┴────────────────────────────┐
                                  ▼                                                         ▼
                       ResNet-18 (full fine-tune)                               ViT-B/16 (full fine-tune)
                       vape probability: P(ResNet)                              vape probability: P(ViT)
                                  │                                                         │
                                  └────────────────────────────┬────────────────────────────┘
                                                               ▼
                                                  verdict triage engine
                                              ┌─────────────────────────────────┐
                                              │ Both >= threshold  ──▶ BLOCKED  │
                                              │ Disagree           ──▶ REVIEW   │
                                              │ Both < threshold   ──▶ ALLOWED  │
                                              └─────────────────────────────────┘
                                                               │
                           Gradio UI: live confidence bars · agreement indicator · portal review
```

## The machine learning  (`notebooks/evaporiser_classifier_v2.ipynb`)
- **Backbones:** 
  - `torchvision.models.resnet18` — lightweight CNN (11.7M params), pretrained on ImageNet-1k.
  - `google/vit-base-patch16-224` — Vision Transformer (86M params) via Hugging Face.
- **Head Adaptation:** Pretrained 1,000-class classifier heads replaced with a fresh 2-class
  binary head (`512 -> 2` on ResNet; `768 -> 2` on ViT). A thin `ViTWrapper` normalises output
  logits so both backbones share an identical training and evaluation loop.
- **Augmentation (every transform justified):** RandomResizedCrop (scale 0.7–1.0) for variable
  seller framing, RandomHorizontalFlip for arbitrary vape orientation, rotation ($\pm 15^\circ$)
  for tilted handheld phone shots, and ColourJitter for indoor white-balance variation.
- **Honest evaluation (zero test-set leakage):** File-level stratified 70/15/15 split seeded
  across all libraries (`SEED 42`). Champion models and decision thresholds were selected
  **strictly on validation data**; the held-out test set was evaluated exactly once for final
  reporting.

### Held-out Test Results

| Configuration | Val Acc | Test Acc | Vape Precision | Vape Recall | Vape F1 | Architecture / Strategy |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`vit_full`** ★ | **0.996** | **0.992** | **0.992** | **0.992** | **0.992** | ViT-B/16 · Full fine-tuning (Winner) |
| `resnet_full` | 0.996 | 0.988 | 0.983 | 0.992 | 0.988 | ResNet-18 · Full fine-tuning (0.996 acc at 0.74 threshold) |
| `vit_head` | 0.992 | 0.988 | 0.992 | 0.983 | 0.987 | ViT-B/16 · Frozen backbone |
| `resnet_head` | 0.967 | 0.971 | 0.952 | 0.992 | 0.971 | ResNet-18 · Frozen backbone |

## Run it
Requires **Python 3.10 – 3.12**.
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

- **Model weights** aren't committed to the repository (too large for Git). Produce them by
  running `notebooks/evaporiser_classifier_v2.ipynb` in Google Colab (takes ~5 minutes on a T4 GPU),
  then place the exported `best_models/` checkpoints into a local `models/` directory.
- **Interactive demonstration mode:** If launched without `.pt` checkpoints, `app.py`
  automatically launches in interactive demonstration mode so the interface and triage engine can
  be inspected immediately.
- **Sample images:** Bundled in `demo_samples/pos/` (real vape listings) and `demo_samples/neg/`
  (lookalikes such as pens and flashlights).

## Project structure
| Path | Role |
|------|------|
| `app.py` | Standalone Gradio seller portal application with triage verdict engine |
| `classifier.py` | PyTorch inference pipeline, `ViTWrapper`, model loader, and preprocessing |
| `notebooks/` | Main training benchmark (`evaporiser_classifier_v2.ipynb`), Colab webapp, and executed PDF results |
| `demo_samples/` | Sample positive (vape) and negative (lookalike) test images for UI evaluation |
| `requirements.txt` | Python dependencies |

## Tech
Python · PyTorch · Torchvision · Hugging Face Transformers · Gradio · Weights & Biases · Pillow · NumPy

## Limitations & further work
- Dataset uses isolated single-product images; real marketplace listings often contain complex
  cluttered backgrounds or multiple bundled accessories.
- Binary classification flags the entire image; object detection (e.g. YOLO) or segmentation
  would provide bounding boxes and localize concealed components.
- Multimodal extension combining image tokens with listing metadata (title, price, description)
  to detect evasion keywords (e.g., "juice pods", "aromatherapy sticks").

## License
MIT — see [LICENSE](LICENSE).
