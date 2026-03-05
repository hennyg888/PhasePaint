import gradio as gr
import torch

from single_gen import get_pipe
from config import STEPS, GUIDANCE_SCALE
from utils import decode_imgs



def create_tab():
    """Batch‑generation interface producing a 3x3 grid of outputs."""

    gr.Markdown("### Batch Generation")
    prompt_txt = gr.Textbox(label="Prompt")
    neg_txt = gr.Textbox(label="Negative prompt")
    run_btn = gr.Button("Generate Batch")
    out_gallery = gr.Gallery(label="Results (3x3)", rows=3, columns=3, type="pil", allow_preview=False)

    # callback fired when user clicks an image in the gallery
    def _clicked(evt: gr.SelectData):
        idx = evt.index
        row = idx // 3
        col = idx % 3
        print(f"Clicked grid position ({row},{col})")

    out_gallery.select(_clicked)

    @torch.no_grad()
    def _batch(prompt: str, negative_prompt: str):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = get_pipe()
        guidance = GUIDANCE_SCALE

        # encode prompts once and expand to batch size 9
        prompt_embeds, neg_embeds = pipe.encode_prompt(
            prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )
        prompt_embeds = prompt_embeds.repeat(9, 1, 1)
        neg_embeds = neg_embeds.repeat(9, 1, 1)
        # stack for classifier-free guidance (uncond/cond)
        prompt_embeds = torch.cat([neg_embeds, prompt_embeds], dim=0)

        # prepare batched latents
        latents = torch.randn(
            (9, pipe.unet.in_channels, pipe.unet.sample_size, pipe.unet.sample_size),
            device=device,
            dtype=torch.float16,
        )
        latents = latents * pipe.scheduler.init_noise_sigma

        pipe.scheduler.set_timesteps(STEPS, device=device)
        timesteps = pipe.scheduler.timesteps

        for t in timesteps:
            latent_model_input = torch.cat([latents] * 2)
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance * (noise_pred_text - noise_pred_uncond)
            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        imgs = decode_imgs(latents)
        return imgs

    run_btn.click(_batch, inputs=[prompt_txt, neg_txt], outputs=out_gallery)
