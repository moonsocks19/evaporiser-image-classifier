"""Seller-Portal Web App — Live Two-Model E-Vaporiser Check.

A Gradio web interface simulating an online marketplace trust-and-safety portal.
Upload a listing photo and both champion models score it live (online inference)
with confidence bars and an automated Blocked / Review / Allowed verdict.
"""

import os
import sys
from typing import List, Tuple

import gradio as gr
from PIL import Image
import torch

from classifier import DemoFallbackModel, LoadedModel, load_models

# Visual styling: calm trust-and-safety console palette
CSS = """
:root {
  --ink: #15233B; --muted: #5C6B82; --paper: #F7F8FA; --card: #FFFFFF;
  --line: #E2E6EC; --brand: #2D6CDF; --clear: #1F9D6B; --review: #C9820B; --block: #C2453C;
}
.gradio-container { background: var(--paper) !important; font-family: 'Inter', system-ui, sans-serif; }
#hero {
  display: flex; align-items: center; gap: 14px; padding: 18px 22px; margin-bottom: 8px;
  background: var(--ink); border-radius: 14px; color: #fff;
}
#hero .dot {
  width: 38px; height: 38px; border-radius: 10px; background: var(--brand);
  display: flex; align-items: center; justify-content: center; font-size: 20px;
}
#hero h1 { font-size: 19px; margin: 0; font-weight: 650; letter-spacing: .2px; }
#hero p { margin: 3px 0 0; font-size: 13px; color: #AEBAD0; }
.panel { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 20px; }
.eyebrow {
  font-size: 11px; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;
  color: var(--muted); margin-bottom: 12px;
}
.verdict { border-radius: 12px; padding: 16px 18px; margin-bottom: 14px; border: 1px solid transparent; }
.verdict h2 { margin: 0 0 4px; font-size: 16px; color: inherit !important; }
.verdict p { margin: 0; font-size: 13px; opacity: .92; color: inherit !important; }
.verdict.allowed { background: #E9F7F1; border-color: #BFE7D6; color: #0E5C3F; }
.verdict.review  { background: #FBF1DC; border-color: #F0DCA8; color: #7A5206; }
.verdict.blocked { background: #FAE9E7; border-color: #F0C5C0; color: #7E2A24; }
.modelcard { border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; margin-bottom: 12px; }
.modelcard .top { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; }
.modelcard .nm { font-weight: 650; color: var(--ink); font-size: 14px; }
.modelcard .acc { font-size: 11.5px; color: var(--muted); }
.bar { height: 9px; border-radius: 6px; background: #EEF1F5; overflow: hidden; }
.bar > span { display: block; height: 100%; transition: width 0.3s ease; }
.barrow { display: flex; justify-content: space-between; font-size: 12px; color: var(--muted); margin: 9px 0 4px; }
.barrow b { color: var(--ink); }
.hint { font-size: 12px; color: var(--muted); line-height: 1.5; }
footer { display: none !important; }
"""

HERO = """
<div id='hero'>
  <div class='dot'>🛡️</div>
  <div>
    <h1>Marketplace Listing Screening Portal</h1>
    <p>Prohibited-item detection · e-vaporiser screening · live dual-model inference</p>
  </div>
</div>
"""


def verdict(vape_probs: List[float], threshold: float) -> Tuple[str, str, str]:
    """Combine the two models' probabilities into a single moderation verdict."""
    flags = [p >= threshold for p in vape_probs]
    if all(flags):
        return (
            "blocked",
            "Blocked — e-vaporiser detected",
            "Both models flagged this photo above threshold. The listing cannot be published.",
        )
    if any(flags):
        return (
            "review",
            "Needs review — possible lookalike",
            "The models disagree. Routed to a human moderator for verification.",
        )
    return (
        "allowed",
        "Looks fine — no e-vaporiser detected",
        "Both models cleared this photo. The listing is approved for publishing.",
    )


def build_interface(models):
    def render(img, threshold):
        if img is None:
            return (
                "<div class='panel'><div class='eyebrow'>Listing review</div>"
                "<p class='hint'>Upload or select a product photo to run moderation check.</p></div>"
            )

        results = [m.predict(img) for m in models]
        vape_probs = [r[1] for r in results]
        state, title, detail = verdict(vape_probs, threshold)

        cards = ""
        for m, (scores, vp) in zip(models, results):
            conf = max(scores.values())
            top = max(scores, key=scores.get)
            bar_color = "var(--block)" if vp >= threshold else "var(--clear)"
            cards += f"""
            <div class='modelcard'>
              <div class='top'>
                <span class='nm'>{m.pretty_name}</span>
                <span class='acc'>{m.name}</span>
              </div>
              <div class='barrow'><span>vape probability</span><b>{vp*100:.1f}%</b></div>
              <div class='bar'><span style='width:{min(100.0, max(0.0, vp*100)):.1f}%;background:{bar_color}'></span></div>
              <div class='barrow' style='margin-top:8px;'>
                <span>top label</span><b>{top} · {conf*100:.1f}% confidence</b></div>
            </div>"""

        agree = "agree" if (vape_probs[0] >= threshold) == (vape_probs[1] >= threshold) else "disagree"
        return f"""
        <div class='panel'>
          <div class='eyebrow'>Listing review</div>
          <div class='verdict {state}'><h2>{title}</h2><p>{detail}</p></div>
          {cards}
          <p class='hint'>The two models <b style='color:var(--review);font-weight:700'>{agree}</b> at the current threshold
          ({threshold*100:.0f}%). Lower the threshold to catch more vapes
          (higher recall); raise it to reduce false alarms on benign listings.</p>
        </div>"""

    # Collect sample images if available
    sample_images = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pos_dir = os.path.join(base_dir, "demo_samples", "pos")
    neg_dir = os.path.join(base_dir, "demo_samples", "neg")

    if os.path.isdir(pos_dir):
        sample_images.extend([os.path.join(pos_dir, f) for f in sorted(os.listdir(pos_dir)) if not f.startswith(".")])
    if os.path.isdir(neg_dir):
        sample_images.extend([os.path.join(neg_dir, f) for f in sorted(os.listdir(neg_dir)) if not f.startswith(".")])

    with gr.Blocks(css=CSS, title="Marketplace Listing Check") as demo:
        gr.HTML(HERO)
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<div class='eyebrow'>New listing</div>")
                gr.Textbox(label="Listing title", placeholder="e.g. Rechargeable handheld accessory", value="")
                gr.Textbox(label="Price", placeholder="$0.00", value="")
                photo = gr.Image(type="pil", label="Product photo", sources=["upload", "clipboard"], height=300)
                threshold = gr.Slider(0.05, 0.95, value=0.5, step=0.05, label="Block threshold (vape probability)")
                gr.HTML("<p class='hint'>Trust & Safety policy: photos likely to show an e-vaporiser are blocked before going live.</p>")

                if sample_images:
                    gr.Examples(
                        examples=[[img, 0.5] for img in sample_images[:6]],
                        inputs=[photo, threshold],
                        label="Sample Test Images (Vapes & Lookalikes)",
                    )

            with gr.Column(scale=1):
                review = gr.HTML("<div class='panel'><div class='eyebrow'>Listing review</div><p class='hint'>Add a product photo to run the check.</p></div>")

        photo.change(render, [photo, threshold], review)
        threshold.change(render, [photo, threshold], review)

    return demo


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = os.getenv("MODELS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))

    print(f"Loading models on {device} from {models_dir}...")
    models = load_models(models_dir, device)

    if not models:
        print("\n[NOTE] No saved .pt weights found in models/ directory.")
        print("To generate production weights:")
        print("  1. Run notebooks/evaporiser_classifier_v2.ipynb in Google Colab (with GPU).")
        print("  2. Place the exported 'best_models/' checkpoints into models/.")
        print("Starting web interface in interactive demonstration mode...\n")
        models = [
            DemoFallbackModel("vit_full", "vit", "full"),
            DemoFallbackModel("resnet_full", "resnet", "full"),
        ]

    for m in models:
        print(f"  ready: {m.name} -> {m.pretty_name}")

    demo = build_interface(models)
    demo.launch(server_port=7860, share=False)


if __name__ == "__main__":
    main()
