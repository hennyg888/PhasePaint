import gradio as gr
import torch

from single_gen import get_pipe
from utils import preview_imgs, decode_imgs, display_image


def create_tab():
    """PhasePaint-specific generation interface.

    Uses the same `LatentRefinerPipeline` but runs it repeatedly
    from a random latent, emitting preview images every
    `STEP_INTERVAL` steps.
    """

    gr.Markdown("### PhasePaint Generation")

    from config import STEPS, GUIDANCE_SCALE, STEP_INTERVAL, START_STEP

    prompt_txt = gr.Textbox(label="Prompt", placeholder="Description of scene")
    neg_txt = gr.Textbox(label="Negative prompt", placeholder="Things to avoid")
    go_btn = gr.Button("Generate/Continue")
    status_slider = gr.Slider(minimum=0, maximum=STEPS, value=0, step=1, label="Iterations completed", interactive=False)
    out_gallery = gr.Gallery(label="Results (3x3)", rows=3, columns=3, type="pil", allow_preview=False)
    
    # persistent state between clicks
    state = gr.State({
        "latents": None,
        "prompt_embeds": None,
        "current": None,
        "guidance": None,
    })

    @torch.no_grad()
    def _step(prompt: str, negative_prompt: str, state: dict):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = get_pipe()

        # initialize state on first click or when completed previously
        if state["latents"] is None:
            guidance = GUIDANCE_SCALE
            # compute embeddings and expand
            prompt_embeds, neg_embeds = pipe.encode_prompt(
                prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = prompt_embeds.repeat(9, 1, 1)
            neg_embeds = neg_embeds.repeat(9, 1, 1)
            prompt_embeds = torch.cat([neg_embeds, prompt_embeds], dim=0)

            # initial random latents
            latents = torch.randn(
                (9, pipe.unet.in_channels, pipe.unet.sample_size, pipe.unet.sample_size),
                device=device,
                dtype=torch.float16,
            )
            latents = latents * pipe.scheduler.init_noise_sigma

            latents = pipe(
                latents=latents,
                prompt_embeds=prompt_embeds,
                start_steps_list=[0] * 9,
                num_inference_steps=[STEPS] * 9,
                guidance_scale=guidance,
                phase_len=START_STEP,
            )
            
            state.update({
                "latents": latents,
                "prompt_embeds": prompt_embeds,
                "current": START_STEP,
                "guidance": guidance,
            })

        # perform one chunk of STEP_INTERVAL steps
        latents = state["latents"]
        prompt_embeds = state["prompt_embeds"]
        guidance = state["guidance"]
        current = state["current"]

        # run the pipeline from current for another interval
        latents = pipe(
            latents=latents,
            prompt_embeds=prompt_embeds,
            start_steps_list=[current] * 9,
            num_inference_steps=[STEPS] * 9,
            guidance_scale=guidance,
            phase_len=STEP_INTERVAL,
        )
        current += STEP_INTERVAL
        state["latents"] = latents
        state["current"] = current

        previews = preview_imgs(latents)
        display_image(previews[0], title=f"PhasePaint preview at {current} steps")

        if current >= STEPS:
            # finished, return final batch and clear state
            final = decode_imgs(latents)
            state.update({"latents": None, "prompt_embeds": None, "current": None, "guidance": None})
            return final, state, str(current)
        else:
            return previews, state, str(current)

    go_btn.click(
        _step,
        inputs=[prompt_txt, neg_txt, state],
        outputs=[out_gallery, state, status_slider],
        queue=True,
    )
