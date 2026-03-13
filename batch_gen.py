import gradio as gr
import torch
from PIL import ImageDraw

from single_gen import get_pipe
from config import STEPS, GUIDANCE_SCALE, GALLERY_SIZE
from utils import decode_imgs, write_image
from logger import log
import time


def create_tab(prompt_txt: gr.components.Textbox, neg_txt: gr.components.Textbox):
    """Batch-generation interface producing a 3x3 grid of outputs.

    The `prompt_txt` and `neg_txt` components are shared across all
    tabs so that changing the text in one tab reflects everywhere.
    """

    gr.Markdown("### Batch Generation")
    run_btn = gr.Button("Generate Batch")
    out_gallery = gr.Gallery(label="Results (3x3)", rows=3, columns=3, type="pil", allow_preview=False, height=GALLERY_SIZE, elem_id="my_gallery")
    save_btn = gr.Button("Save Selected Images", interactive=False)

    state = gr.State({
        "images": [],  # most recent batch of decoded images
        "selected": [],  # indices of currently selected images
    })

    # selection state and helper functions
    def draw_border(img):
        img = img.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size
        thickness = 10
        for i in range(thickness):
            draw.rectangle([i, i, w-i-1, h-i-1], outline="red")
        return img

    def _toggle_select(evt: gr.SelectData, state):
        log("[batch_gen] gallery image selected")
        selected = state.get("selected")
        if selected is None:
            selected = []

        idx = evt.index
        if idx in selected:
            selected.remove(idx)
        else:
            selected.append(idx)
        state["selected"] = selected

        # rebuild gallery with border on selected
        gallery_items = []
        images = state.get("images", [])
        for i, img in enumerate(images):
            if i in selected:
                gallery_items.append(draw_border(img))
            else:
                gallery_items.append(img)
        gallery = gr.Gallery(value=gallery_items, selected_index=None)
        return gallery, state

    out_gallery.select(_toggle_select, inputs=state, outputs=[out_gallery, state])

    def _save_images(state: dict):
        log("[batch_gen] save button clicked")
        # save selected with 'saved' tag, others as 'discarded'
        imgs = state.get("images", [])
        sel = state.get("selected", [])
        for idx, img in enumerate(imgs):
            tag = "saved" if idx in sel else "discarded"
            write_image(img, "batch_gen", tag=tag, idx=idx)
        # reset state after write
        state["selected"] = []
        state["images"] = []
        return [], state, gr.Button(interactive=True), gr.Button(interactive=False)

    save_btn.click(_save_images, inputs=state, outputs=[out_gallery, state, run_btn, save_btn])


    @torch.no_grad()
    def _batch(prompt: str, negative_prompt: str, state: dict = None):
        log("[batch_gen] generate button clicked")
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
        # store previews in state so selection can highlight
        if state is not None:
            state["images"] = imgs
        return imgs, state, gr.Button(interactive=False), gr.Button(interactive=True)

    run_btn.click(_batch, inputs=[prompt_txt, neg_txt, state], outputs=[out_gallery, state, run_btn, save_btn])
