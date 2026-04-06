import gradio as gr
import torch
from PIL import ImageDraw

from single_gen import get_pipe
from utils import preview_imgs, decode_imgs, write_image
from logger import log
from config import NEW_FEATS

def draw_green_border(img):
    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    thickness = 20
    for i in range(thickness):
        draw.rectangle([i, i, w-i-1, h-i-1], outline="green")
    return img

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

def toggle_select(evt: gr.SelectData, state, duplicate_mode=False):
    log("[PhasePaint_gen] gallery image selected")
    selected = state.get("selected") or []
    duplicate = state.get("duplicate") or []

    idx = evt.index

    if duplicate_mode:
        if idx in duplicate:
            duplicate.remove(idx)
        else:
            if idx in selected:
                selected.remove(idx)
            duplicate.append(idx)
    else:
        if idx in selected:
            selected.remove(idx)
        else:
            if idx in duplicate:
                duplicate.remove(idx)
            selected.append(idx)

    state["selected"] = selected
    state["duplicate"] = duplicate

    # rebuild gallery with class applied
    gallery_items = []
    images = state.get("previews") or []
    for i, img in enumerate(images):
        if i in duplicate:
            gallery_items.append(draw_green_border(img))
        elif i in selected:
            gallery_items.append(draw_cross(img))
        else:
            gallery_items.append(img)
    gallery = gr.Gallery(value=gallery_items, selected_index=None)
    return gallery, state

def create_tab(prompt_txt: gr.components.Textbox, neg_txt: gr.components.Textbox, total_iterations: gr.components.Number | None = None):
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
    status_slider = gr.Slider(minimum=0, maximum=STEPS, value=0, step=1, label="Iterations Completed", interactive=False)

    duplicate_mode = None
    first_phase_len = None
    phase_count = None
    if NEW_FEATS:
        default_phase_count = 1 + max(0, (STEPS - START_STEP + STEP_INTERVAL - 1) // STEP_INTERVAL)
        with gr.Row():
            duplicate_mode = gr.Checkbox(label="Duplicate on click", value=False)
            first_phase_len = gr.Number(label="First phase length", value=START_STEP, precision=0, step=1)
            phase_count = gr.Number(label="Number of phases", value=default_phase_count, precision=0, step=1)
            preset_dropdown = gr.Dropdown(
                choices=["Custom", "Draft Mode", "Standard Mode", "Long Mode"],
                value="Custom",
                label="Presets",
            )

        def _apply_preset(preset):
            updates = []
            if total_iterations is not None:
                if preset == "Draft Mode":
                    updates.append(gr.update(value=25))
                elif preset == "Standard Mode":
                    updates.append(gr.update(value=50))
                elif preset == "Long Mode":
                    updates.append(gr.update(value=50))
                else:
                    updates.append(gr.update())
            if preset == "Draft Mode":
                updates.append(gr.update(value=15))
                updates.append(gr.update(value=2))
            elif preset == "Standard Mode":
                updates.append(gr.update(value=20))
                updates.append(gr.update(value=4))
            elif preset == "Long Mode":
                updates.append(gr.update(value=35))
                updates.append(gr.update(value=3))
            else:
                updates.append(gr.update())
                updates.append(gr.update())
            return tuple(updates)

        preset_outputs = []
        if total_iterations is not None:
            preset_outputs.append(total_iterations)
        preset_outputs.extend([first_phase_len, phase_count])
        preset_dropdown.change(
            _apply_preset,
            inputs=preset_dropdown,
            outputs=preset_outputs,
        )

        def _update_status_max(value):
            total = STEPS
            if value is not None:
                total = max(1, int(value))
            return gr.update(maximum=total)
        if total_iterations is not None:
            total_iterations.change(
                _update_status_max,
                inputs=total_iterations,
                outputs=status_slider,
            )

    out_gallery = gr.Gallery(label="Results (3x3)", rows=3, columns=3, type="pil", allow_preview=False, height=GALLERY_SIZE, elem_id="my_gallery")   
    save_btn = gr.Button("Save Images", interactive=False)

   

    state = gr.State({
        "latents": None,
        "prompt_embeds": None,
        "current": None,
        "guidance": None,
        "selected": [],
        "duplicate": [],
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
        return [], state, str(0), gr.Button(interactive=True), gr.Button(interactive=False)

    save_btn.click(_save_images, inputs=state, outputs=[out_gallery, state, status_slider, go_btn, save_btn])

    select_inputs = [state]
    if duplicate_mode is not None:
        select_inputs.append(duplicate_mode)

    out_gallery.select(
        toggle_select,
        inputs=select_inputs,
        outputs=[out_gallery, state]
    )

    @torch.no_grad()
    def _step(prompt: str, negative_prompt: str, state: dict = None, total_iterations=50, first_phase_len_value=20, phase_count_value=4):
        log("[PhasePaint_gen] Generate/Continue button clicked")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = get_pipe()

        total_steps = STEPS
        if total_iterations is not None:
            total_steps = max(1, int(total_iterations))

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

            first_phase_len = min(START_STEP, total_steps)
            if first_phase_len_value is not None:
                requested_first_phase = max(1, int(first_phase_len_value))
                if requested_first_phase <= total_steps:
                    first_phase_len = requested_first_phase

            if phase_count_value is not None:
                requested_phase_count = max(1, int(phase_count_value))
            else:
                requested_phase_count = None

            if requested_phase_count is None:
                total_phases = 1 + max(0, (total_steps - first_phase_len + STEP_INTERVAL - 1) // STEP_INTERVAL)
            else:
                # number_of_phases is lower priority than first_phase_len,
                # so only adjust the schedule if the requested phase count makes sense.
                total_phases = requested_phase_count

            if total_phases <= 1 or first_phase_len >= total_steps:
                first_phase_len = total_steps
                total_phases = 1
            else:
                first_phase_len = min(first_phase_len, total_steps)

            latents = pipe(
                latents=latents,
                prompt_embeds=prompt_embeds,
                start_steps_list=[0] * 9,
                num_inference_steps=[total_steps] * 9,
                guidance_scale=guidance,
                phase_len=first_phase_len,
            )
            
            state.update({
                "latents": latents,
                "prompt_embeds": prompt_embeds,
                "current": first_phase_len,
                "guidance": guidance,
                "phase_len": first_phase_len,
                "phase_count": total_phases,
            })

            if first_phase_len >= total_steps:
                final = decode_imgs(latents)
                state["previews"] = final
                state.update({"latents": None, "prompt_embeds": None, "current": None, "guidance": None})
                return final, state, str(first_phase_len), gr.Button(interactive=False), gr.Button(interactive=True)

            previews = preview_imgs(latents)
            state["previews"] = previews
            return previews, state, str(first_phase_len), gr.skip(), gr.skip()

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
        duplicate = state.get("duplicate", []) or []
        original_latents = latents
        original_prompt_embeds = prompt_embeds

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

        # handle duplicate selections when feature is enabled
        if duplicate:
            batch_size = original_latents.shape[0]
            unique_duplicates = []
            for idx in sorted(set(duplicate)):
                if idx in selected:
                    continue
                if 0 <= idx < batch_size:
                    unique_duplicates.append(idx)

            available_slots = 9 - latents.shape[0]
            dup_indices = unique_duplicates[:available_slots]
            if dup_indices:
                dup_latents = original_latents[dup_indices].clone()
                neg_embeds = original_prompt_embeds[:batch_size][dup_indices]
                pos_embeds = original_prompt_embeds[batch_size:][dup_indices]
                dup_prompt_embeds = torch.cat([neg_embeds, pos_embeds], dim=0)
                latents = torch.cat([latents, dup_latents], dim=0)
                prompt_embeds = torch.cat([prompt_embeds, dup_prompt_embeds], dim=0)
            state["duplicate"] = []

        # if nothing left to process, we're effectively done
        if latents.numel() == 0:
            # nothing to denoise; build an empty output and clear state
            state.update({"latents": None, "prompt_embeds": None,
                          "current": None, "guidance": None})
            return [], state, str(0), gr.skip()

        # run the pipeline from current for another interval; lists size
        # should match the current batch
        n = latents.shape[0]
        phase_len = state.get("phase_len", START_STEP)
        phase_count = state.get("phase_count", 1)
        remaining_after_first = max(0, total_steps - phase_len)
        if phase_count <= 1 or remaining_after_first <= 0:
            next_phase_len = max(1, total_steps - current)
        else:
            phases_left = max(1, phase_count - 1)
            next_phase_len = (remaining_after_first + phases_left - 1) // phases_left
            next_phase_len = min(next_phase_len, max(1, total_steps - current))

        latents = pipe(
            latents=latents,
            prompt_embeds=prompt_embeds,
            start_steps_list=[current] * n,
            num_inference_steps=[total_steps] * n,
            guidance_scale=guidance,
            phase_len=next_phase_len,
        )

        current += next_phase_len
        state["latents"] = latents
        state["prompt_embeds"] = prompt_embeds
        state["current"] = current

        if current >= total_steps:
            # finished, decode and save all remaining images as "saved".
            final = decode_imgs(latents)
            state["previews"] = final
            state.update({"latents": None, "prompt_embeds": None, "current": None, "guidance": None})
            return final, state, str(current), gr.Button(interactive=False), gr.Button(interactive=True)
        else:
            previews = preview_imgs(latents)
            state["previews"] = previews
            return previews, state, str(current), gr.skip(), gr.skip()

    inputs = [prompt_txt, neg_txt]
    inputs.append(state)
    if total_iterations is not None:
        inputs.append(total_iterations)
    if first_phase_len is not None:
        inputs.append(first_phase_len)
        inputs.append(phase_count)

    go_btn.click(
        _step,
        inputs=inputs,
        outputs=[out_gallery, state, status_slider, go_btn, save_btn],
        queue=True,
    )
