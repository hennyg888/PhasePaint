import gradio as gr
import torch

from refiner_pipe import LatentRefinerPipeline
from utils import preview_imgs, decode_imgs

# create pipeline once and reuse; pipelines are heavy to instantiate
_GLOBAL_PIPE: LatentRefinerPipeline | None = None

def get_pipe() -> LatentRefinerPipeline:
    global _GLOBAL_PIPE
    if _GLOBAL_PIPE is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _GLOBAL_PIPE = LatentRefinerPipeline()
        # ensure model moved to device if pipeline constructor didn't
        _GLOBAL_PIPE.to(device)
        print("loading pipeline...")
    return _GLOBAL_PIPE


def create_tab():
    """Builds the components for the "single gen" tab.

    This function is intended to be called from an outer
    `gr.Blocks` context (for example, within a `TabItem`).
    It exposes a prompt/negative-prompt interface and drives a
    diffusion pipeline based on :class:`LatentRefinerPipeline`.
    """

    gr.Markdown("### Single Generation")

    prompt_txt = gr.Textbox(label="Prompt", placeholder="Enter a prompt...")
    neg_txt = gr.Textbox(label="Negative prompt", placeholder="Things to avoid")

    out_img = gr.Image(type="pil", label="Result")
    generate_btn = gr.Button("Generate")
    cancel_btn = gr.Button("Cancel")

    # simple mutable flag for cancellation
    cancel_requested = {"flag": False}

    def _request_cancel():
        cancel_requested["flag"] = True

    cancel_btn.click(_request_cancel)

    # generator function will yield intermediate previews
    # import the fixed parameters lazily to avoid circular import issues
    from config import STEPS, GUIDANCE_SCALE, PREVIEW_INTERVAL

    @torch.no_grad()
    def _generate(prompt: str, negative_prompt: str):
        # reset flag every time we start
        cancel_requested["flag"] = False
        # build pipeline & embeddings (reusing global instance)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = get_pipe()
        guidance = GUIDANCE_SCALE

        # encode prompt(s)
        prompt_embeds, neg_embeds = pipe.encode_prompt(
            prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative_prompt,
        )
        prompt_embeds=torch.cat([neg_embeds, prompt_embeds], dim=0)

        # prepare latents
        latents = torch.randn(
            (1, pipe.unet.in_channels, pipe.unet.sample_size, pipe.unet.sample_size),
            device=device, dtype=torch.float16
        )
        latents = latents * pipe.scheduler.init_noise_sigma

        # configure scheduler timesteps for full run
        pipe.scheduler.set_timesteps(STEPS, device=device)
        timesteps = pipe.scheduler.timesteps

        preview_interval = PREVIEW_INTERVAL

        for i, t in enumerate(timesteps):
            # check cancellation flag
            if cancel_requested["flag"]:
                # clear flag so next run starts fresh
                cancel_requested["flag"] = False
                return

            # two forward passes for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            #print("latent_model_input.dtype:", latent_model_input.dtype)
            #print("prompt_embeds.dtype:", prompt_embeds.dtype)
            #print("t.dtype:", t.dtype)
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance * (noise_pred_text - noise_pred_uncond)

            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

            # every so often show a preview
            if (i + 1) % preview_interval == 0 or i == len(timesteps) - 1:
                preview = preview_imgs(latents)[0]
                #display_image(preview, title=f"Preview step {i+1}/{STEPS}")
                print(f"Yielding preview at step {i+1}/{STEPS}")
                yield preview

        # final decode
        final = decode_imgs(latents)[0]
        # make sure flag cleared
        cancel_requested["flag"] = False
        yield final

    # use a queued function so that events run sequentially
    generate_btn.click(
        _generate,
        inputs=[prompt_txt, neg_txt],
        outputs=out_img,
        queue=True,
    )
