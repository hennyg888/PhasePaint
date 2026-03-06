import torch
import os
from datetime import datetime

import matplotlib.pyplot as plt
def display_image(image, title="Generated Image"):
    """
    Display image in Jupyter notebook with title
    
    Args:
        image (PIL.Image): Image to display
        title (str): Title for the image
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)
    plt.show()


def write_image(img, gen_type: str, tag: str | None = None, user: str | None = None, idx: int | None = None):
    """Save an image file with a standardized filename.

    The generated filename includes the user identifier, the generation
    type (e.g. ``single_gen`` or ``batch_gen``), an optional tag, and a
    timestamp. This helper centralizes the logic so both single and batch
    modules can reuse it.

    Args:
        img (PIL.Image): Image to save.
        gen_type (str): Short descriptor of where the image originated.
        tag (str|None): Optional extra descriptor ("saved", "cancelled",
            etc.).
        user (str|None): Username to include; if None the value from
            :mod:`config` is used.
        idx (int|None): Optional index to include in the filename.

    Returns:
        str: Path where the image was written.
    """
    if user is None:
        from config import USER
        user = USER

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_part = f"_{tag}" if tag else ""
    idx_part = f"_{idx}" if idx is not None else ""
    fname = f"{user}_{gen_type}{tag_part}{idx_part}_{ts}.png"
    os.makedirs("saved_imgs", exist_ok=True)
    path = os.path.join("saved_imgs", fname)
    img.save(path)
    print(f"image saved to {path}")
    return path

from diffusers import AutoencoderKL, AutoencoderTiny 
from diffusers.image_processor import VaeImageProcessor

preview_vae = AutoencoderTiny.from_pretrained("cqyan/hybrid-sd-tinyvae", torch_dtype=torch.float16, low_cpu_mem_usage=False, device_map=None).to("cuda")
full_vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae", torch_dtype=torch.float16).to("cuda")

vae_scale_factor = 2 ** (len(full_vae.config.block_out_channels) - 1)
image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

@torch.no_grad()
def preview_imgs(latents):
    # Scale latents back to the VAE's expected range
    latents_for_decoding = latents / preview_vae.config.scaling_factor
    
    # Decode the latents
    images = preview_vae.decode(latents_for_decoding).sample
    
    # Convert to PIL image
    images = image_processor.postprocess(images, output_type='pil')
    return images

@torch.no_grad()
def decode_imgs(latents):
    # Scale latents back to the VAE's expected range
    latents_for_decoding = latents / full_vae.config.scaling_factor
    
    # Decode the latents
    images = full_vae.decode(latents_for_decoding).sample
    
    # Convert to PIL image
    images = image_processor.postprocess(images, output_type='pil')
    return images

# Source - https://stackoverflow.com/a
# Posted by y.selivonchyk, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-15, License - CC BY-SA 4.0
import subprocess as sp

def get_total_vram():
    command = "nvidia-smi --query-gpu=memory.total --format=csv"
    total_mem_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    total_mem = [int(x.split()[0]) for i, x in enumerate(total_mem_info)][0]
    return total_mem


def get_free_vram():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_value = [int(x.split()[0]) for i, x in enumerate(memory_free_info)][0]
    return memory_free_value