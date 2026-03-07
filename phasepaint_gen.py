import gradio as gr
import torch
from PIL import ImageDraw

from single_gen import get_pipe
from utils import preview_imgs, decode_imgs, write_image
from logger import log

def draw_cross(img):
    # draw a solid red 'X' over the image instead of a border
    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    line_width = max(1, min(w, h) // 20)
    # diagonal from top-left to bottom-right
    draw.line((0, 0, w, h), fill="red", width=line_width)
    # diagonal from top-right to bottom-left
    draw.line((w, 0, 0, h), fill="red", width=line_width)
    return img

def toggle_select(evt: gr.SelectData, state):
    log("[PhasePaint_gen] gallery image selected")
    selected = state["selected"]
    if selected is None:
        selected = []

    idx = evt.index

    if idx in selected:
        selected.remove(idx)
    else:
        selected.append(idx)
    state["selected"] = selected

    # rebuild gallery with class applied
    gallery_items = []
    images = state["previews"]
    for i, img in enumerate(images):
        if i in selected:
            gallery_items.append(draw_cross(img))
        else:
            gallery_items.append(img)
    gallery = gr.Gallery(value=gallery_items, selected_index=None)
    return gallery, state

def create_tab(prompt_txt: gr.components.Textbox, neg_txt: gr.components.Textbox):
    """PhasePaint-specific generation interface.

    Uses the same `LatentRefinerPipeline` but runs it repeatedly
    from a random latent, emitting preview images every
    `STEP_INTERVAL` steps.  The prompt textboxes are shared across
    tabs, so the user sees the same content no matter where they
    type.
    """

    gr.Markdown("### PhasePaint Generation")

    from config import STEPS, GUIDANCE_SCALE, STEP_INTERVAL, START_STEP, GALLERY_SIZE

    go_btn = gr.Button("Generate/Continue")
    status_slider = gr.Slider(minimum=0, maximum=STEPS, value=0, step=1, label="Iterations completed", interactive=False)
    out_gallery = gr.Gallery(label="Results (3x3)", rows=3, columns=3, type="pil", allow_preview=False, height=GALLERY_SIZE, elem_id="my_gallery")   
    save_btn = gr.Button("Save Images")

    state = gr.State({
        "latents": None,
        "prompt_embeds": None,
        "current": None,
        "guidance": None,
        "selected": [],
        "previews": None,
    })

    def _save_images(state: dict):
        if state.get("current") is not None or state.get("previews") is None:
            return state  # still generating; ignore save clicks
        log("[PhasePaint_gen] save button clicked")
        # save selected with 'saved' tag, others as 'discarded'
        imgs = state.get("previews", [])
        sel = state.get("selected", [])
        for idx, img in enumerate(imgs):
            tag = f"discarded_itr-{STEPS}" if idx in sel else "saved"
            write_image(img, "PhasePaint_gen", tag=tag, idx=idx)
        # reset state after write
        state["selected"] = []
        state["previews"] = []
        return [], state, str(0), gr.Button(interactive=True)

    save_btn.click(_save_images, inputs=state, outputs=[out_gallery, state, status_slider, go_btn])

    out_gallery.select(
        toggle_select,
        inputs=state,
        outputs=[out_gallery, state]
    )

    @torch.no_grad()
    def _step(prompt: str, negative_prompt: str, state: dict):
        log("[PhasePaint_gen] Generate/Continue button clicked")
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
                "guidance": guidance
            })

        # perform one chunk of STEP_INTERVAL steps
        latents = state["latents"]
        prompt_embeds = state["prompt_embeds"]
        guidance = state["guidance"]
        current = state["current"]

        # if the user selected any images, save the discarded previews
        # and then expunge them completely from the optimization batch.
        # this means dropping their latents and associated prompt
        # embeddings so subsequent steps only act on the remaining entries.
        selected = state.get("selected", []) or []
        if selected:
            # save preview versions of any selected images before they are
            # removed. fall back to decoding if previews aren't available.
            previews = state.get("previews")
            if previews is None:
                decoded = decode_imgs(latents)
                previews = decoded
            for idx in selected:
                write_image(previews[idx], "PhasePaint_gen", tag=f"discarded_itr-{current}", idx=idx)


            batch_size = latents.shape[0]
            keep = [i for i in range(batch_size) if i not in selected]
            # filter latents
            latents = latents[keep]
            # prompt_embeds is neg then pos concatenated; each half has
            # batch_size entries. we keep the corresponding slices.
            neg_embeds = prompt_embeds[:batch_size][keep]
            pos_embeds = prompt_embeds[batch_size:][keep]
            prompt_embeds = torch.cat([neg_embeds, pos_embeds], dim=0)
            # also clear selection so we don't try to remove again
            state["selected"] = []

        # if nothing left to process, we're effectively done
        if latents.numel() == 0:
            # nothing to denoise; build an empty output and clear state
            state.update({"latents": None, "prompt_embeds": None,
                          "current": None, "guidance": None})
            return [], state, str(0), gr.skip()

        # run the pipeline from current for another interval; lists size
        # should match the current batch
        n = latents.shape[0]
        latents = pipe(
            latents=latents,
            prompt_embeds=prompt_embeds,
            start_steps_list=[current] * n,
            num_inference_steps=[STEPS] * n,
            guidance_scale=guidance,
            phase_len=STEP_INTERVAL,
        )

        current += STEP_INTERVAL
        state["latents"] = latents
        state["prompt_embeds"] = prompt_embeds
        state["current"] = current

        if current >= STEPS:
            # finished, decode and save all remaining images as "saved".
            final = decode_imgs(latents)
            state["previews"] = final
            state.update({"latents": None, "prompt_embeds": None, "current": None, "guidance": None})
            return final, state, str(current), gr.Button(interactive=False)
        else:
            previews = preview_imgs(latents)
            state["previews"] = previews
            return previews, state, str(current), gr.skip()

    go_btn.click(
        _step,
        inputs=[prompt_txt, neg_txt, state],
        outputs=[out_gallery, state, status_slider, go_btn],
        queue=True,
    )
