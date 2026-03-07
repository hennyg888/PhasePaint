import gradio as gr
import torch
from datetime import datetime
import os

from logger import log

from refiner_pipe import LatentRefinerPipeline
from utils import preview_imgs, decode_imgs, write_image

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


def create_tab(prompt_txt: gr.components.Textbox, neg_txt: gr.components.Textbox):
    """Builds the components for the "single gen" tab.

    The prompt textboxes are supplied by the caller so they can be
    shared across tabs.  Callers should create the boxes once and
    pass them in.
    """

    gr.Markdown("### Single Generation")

    with gr.Row():
        generate_btn = gr.Button("Generate")
        cancel_btn = gr.Button("Cancel")
    out_img = gr.Image(type="pil", label="Result")
    save_btn = gr.Button("Save Image", interactive=False)


    # simple mutable flag for cancellation
    cancel_requested = {"flag": False}

    # store the last successfully completed image (None if none or cancelled)
    last_image = {"img": None}
    # track most recent preview (for cancelled save)
    last_preview = {"img": None}

    def _request_cancel():
        # mark for cancellation and save last preview if available
        log("[single_gen] cancel button pressed")
        if not cancel_requested["flag"]:
            cancel_requested["flag"] = True
        img = last_preview.get("img")
        if img is not None and last_image.get("img") is None:
            write_image(img, "single_gen", tag="cancelled")

    cancel_btn.click(_request_cancel)

    # generator function will yield intermediate previews
    # import the fixed parameters lazily to avoid circular import issues
    from config import STEPS, GUIDANCE_SCALE, PREVIEW_INTERVAL, USER

    @torch.no_grad()
    def _generate(prompt: str, negative_prompt: str):
        # log generate action
        log("[single_gen] generate button pressed")
        # if there was a previous completed image that wasn't saved, mark it discarded
        if last_image.get("img") is not None and not cancel_requested.get("flag"):
            write_image(last_image["img"], "single_gen", tag="discarded")
        # reset flag every time we start
        cancel_requested["flag"] = False
        # clear last image – new generation in progress
        last_image["img"] = None
        last_preview["img"] = None
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
                yield None, gr.skip()
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
                # keep for potential cancel save
                last_preview["img"] = preview
                #display_image(preview, title=f"Preview step {i+1}/{STEPS}")
                #print(f"Yielding preview at step {i+1}/{STEPS}")
                yield preview, gr.skip()

        # final decode
        final = decode_imgs(latents)[0]
        # make sure flag cleared
        cancel_requested["flag"] = False
        # save completed image into state in case user wants to save it
        last_image["img"] = final
        yield final, gr.Button(interactive=True)

    # hook up save button (no outputs needed)
    def _save_image():
        # only save if a final image exists and we are not in cancelled state
        if cancel_requested.get("flag"):
            return
        img = last_image.get("img")
        if img is None:
            return
        log("[single_gen] save button pressed")
        last_image["img"] = None
        write_image(img, "single_gen", tag="saved")
        return gr.Button(interactive=False), None

    save_btn.click(_save_image, outputs=[save_btn, out_img])

    # use a queued function so that events run sequentially
    generate_btn.click(
        _generate,
        inputs=[prompt_txt, neg_txt],
        outputs=[out_img, save_btn]
    )
