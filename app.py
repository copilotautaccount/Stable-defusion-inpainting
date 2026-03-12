import gradio as gr
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
from peft import PeftModel
from pathlib import Path
import os

# --- Configuration ---
PROJECT_ROOT  = Path(__file__).parent
TRAIN_OUTPUT  = PROJECT_ROOT / "outputs" / "interior-inpainting"
LORA_DIR      = str(TRAIN_OUTPUT / "unet_lora")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32

# Read base model from training_metadata.json if available (written by train.py)
import json as _json
_metadata_path = TRAIN_OUTPUT / "training_metadata.json"
if _metadata_path.exists():
    with open(_metadata_path) as _f:
        BASE_MODEL_ID = _json.load(_f).get("base_model", "runwayml/stable-diffusion-inpainting")
    print(f"Base model from training metadata: {BASE_MODEL_ID}")
else:
    BASE_MODEL_ID = "runwayml/stable-diffusion-inpainting"

print("Loading base pipeline...")
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=DTYPE,
    safety_checker=None,
)

if Path(LORA_DIR).exists():
    print(f"Loading LoRA adapter from: {LORA_DIR}")
    pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_DIR)
else:
    print(f"⚠️ LoRA dir not found ({LORA_DIR}), running base model only.")

pipe = pipe.to(DEVICE)
pipe.enable_attention_slicing()
print(f"✅ Pipeline ready on {DEVICE.upper()}")

def process_editor_input(editor_value):
    """
    Gradio ImageEditor returns a dict:
      { "background": PIL.Image, "layers": [PIL.Image, ...], "composite": PIL.Image }
    We use 'background' as the source image and merge all layers into a single mask.
    A white (non-transparent) pixel in any layer counts as the mask region.
    """
    bg = editor_value["background"].convert("RGB").resize((512, 512))

    # Build mask from layer alpha channels
    mask_arr = np.zeros((512, 512), dtype=np.uint8)
    for layer in editor_value.get("layers", []):
        if layer is None:
            continue
        layer_resized = layer.resize((512, 512)).convert("RGBA")
        alpha = np.array(layer_resized)[..., 3]  # alpha channel
        mask_arr = np.maximum(mask_arr, (alpha > 10).astype(np.uint8) * 255)

    mask = Image.fromarray(mask_arr, mode="L")
    return bg, mask

def inpaint(editor_value, prompt, neg_prompt, steps, guidance, strength, seed):
    """Main inpainting function."""
    if editor_value is None or editor_value.get("background") is None:
        return None, "❌ Please upload an image first."

    image, mask = process_editor_input(editor_value)

    if mask.getextrema() == (0, 0):
        return None, "❌ Mask is empty. Please draw on the image to mark the region to edit."

    if not prompt:
        return None, "Prompt is empty. Please provide a prompt."

    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

    output = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt or None,
        image=image,
        mask_image=mask,
        num_inference_steps=int(steps),
        guidance_scale=guidance,
        strength=strength,
        generator=generator,
    ).images[0]

    return output, "✅ Inference complete!"

# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# 🎨 Stable Diffusion Inpainting")
    gr.Markdown("Tải ảnh lên, dùng brush để vẽ vùng cần chỉnh sửa (mask), nhập prompt và chạy inference.")
    
    with gr.Row():
        with gr.Column():
            # ImageEditor: upload image and draw mask with brush
            image_input = gr.ImageEditor(
                sources=["upload"],
                type="pil",
                label="Input Image — Draw mask on the region to edit",
                height=400,
            )
            
            prompt = gr.Textbox(label="Prompt", value="a modern living room with a white sofa and wooden floor")
            neg_prompt = gr.Textbox(label="Negative Prompt", value="blurry, low quality, distorted, ugly, out of focus")

            with gr.Accordion("Advanced Settings", open=False):
                steps = gr.Slider(label="Inference Steps", minimum=10, maximum=100, value=50, step=1)
                guidance = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=20.0, value=7.5, step=0.5)
                strength = gr.Slider(label="Strength", minimum=0.1, maximum=1.0, value=0.99, step=0.01)
                seed = gr.Number(label="Seed", value=42)

            run_button = gr.Button("Run Inpainting", variant="primary")

        with gr.Column():
            result_image = gr.Image(label="Result", height=400)
            status_text = gr.Textbox(label="Status", interactive=False)

    run_button.click(
        fn=inpaint,
        inputs=[image_input, prompt, neg_prompt, steps, guidance, strength, seed],
        outputs=[result_image, status_text],
    )

if __name__ == "__main__":
    demo.launch()
