import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.utils import logging
from collections import defaultdict
from typing import Optional, List

logger = logging.get_logger(__name__)  # For logging

#number of iterations in a phase of denoising
#phase_len = 10
model_id = "runwayml/stable-diffusion-v1-5"

class LatentRefinerPipeline(DiffusionPipeline):
    """
    A custom pipeline that takes a latent and continues the denoising process
    for a specific range of steps.

    This pipeline operates entirely on latents on the GPU and can optionally
    return a quick preview image using a small, fast VAE.
    """

    def __init__(
        self,
        unet: UNet2DConditionModel = None,
        scheduler: DDIMScheduler = None,
        tokenizer: CLIPTokenizer = None,
        text_encoder: CLIPTextModel = None
    ):
        super().__init__()

        if unet is None:
            unet = UNet2DConditionModel.from_pretrained(
                model_id,
                subfolder="unet",
                torch_dtype=torch.float16
            ).to("cuda")
        
        if scheduler is None:
            scheduler =  DDIMScheduler.from_pretrained(
                model_id,
                subfolder="scheduler"
            )

        if tokenizer is None:
            tokenizer = CLIPTokenizer.from_pretrained(
                model_id,
                subfolder="tokenizer"
            )
        
        if text_encoder is None:
            text_encoder = CLIPTextModel.from_pretrained(
                model_id,
                subfolder="text_encoder",
                torch_dtype=torch.float16
            ).to("cuda")

        self.register_modules(unet=unet, scheduler=scheduler, tokenizer=tokenizer, text_encoder=text_encoder)

    @torch.no_grad()
    def __call__(
        self,
        latents: torch.FloatTensor,
        prompt_embeds: torch.FloatTensor,
        start_steps_list: [],   #no more end steps, all end steps are start step + phase_len
        num_inference_steps: [],
        guidance_scale: float = 7.5,
        return_preview_image: bool = False,
        phase_len: int = 10,
    ):
        """
        Args:
            latents (`torch.FloatTensor`):
                The input latent tensor to continue denoising from.
                Should be on the correct device (GPU).
            prompt_embeds (`torch.FloatTensor`):
                The pre-computed text embeddings for the prompt.
            start_step (list):
                The step index to start denoising from (0-indexed). One for each latent.
            num_inference_steps (list):
                The total number of steps in the original diffusion process.
                This is used to calculate the correct timesteps. One for each latent.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in Classifier-Free Diffusion Guidance.
                `guidance_scale` is defined as `w` of equation 2. of
                [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale > 1`.
                Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            preview_vae (`AutoencoderKL`, *optional*):
                A small, fast VAE for generating a preview image.
                If `None`, the main VAE will be used (slower).
            return_preview_image (`bool`, *optional*, defaults to `False`):
                Whether to return a decoded preview image.

        Returns:
            `dict`: A dictionary containing the final `latents` and optionally a `preview_image`.
        """

        device = "cuda"
        latents = latents.to(device)
        #print(type(self.scheduler))

        timesteps_list = []
        # 1. Setup timesteps
        for i, start_step in enumerate(start_steps_list):
            self.scheduler.set_timesteps(num_inference_steps[i], device=device)
            timesteps = self.scheduler.timesteps[start_step:start_step+phase_len]
            timesteps_list.append(timesteps)
        timesteps_list = torch.stack(timesteps_list, dim=0)
        # 2. Denoising loop
        for i in range(phase_len):
            #print("start step ", i)
            # Classifier-free guidance requires two forward passes.
            # We concatenate the unconditional and conditional embeddings.
            latent_model_input = torch.cat([latents] * 2)
            #scaling unneeded for DDIM scheduler
            #latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Predict the noise residual
            # print("start unet")
            #print(prompt_embeds.shape)
            #print(latent_model_input.shape)
            #bad naive fix for prompt_embeds dim not matching latent
            #if i == 1: prompt_embeds = torch.cat([prompt_embeds] * 2)
            timestep_tensor = timesteps_list[:, i]
            dbl_timestep_tensor = torch.cat([timestep_tensor] * 2)
            # print("dbl_timestep_tensor.device: ",dbl_timestep_tensor.device)
            # print("latent_model_input.device: ",latent_model_input.device)
            # print("prompt_embeds.device:",prompt_embeds.device)
            noise_pred = self.unet(latent_model_input, dbl_timestep_tensor, encoder_hidden_states=prompt_embeds).sample

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2, dim=0)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Compute the previous noisy sample x_t -> x_t-1
            #print("Noise pred shape: ",noise_pred.shape)
            #get buckets of indices where the scheduler.step can be batched, must have matching t


            #latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            #print("done step ", i)
            # assume items is a list of (noise_pred, t, latents)
            
            #repeated checking likely unnecessary, if the first iter timestep matches the rest
            #of the iters in the should also match
            batchable_indices = defaultdict(list)

            for idx, val in enumerate(timestep_tensor):
                batchable_indices[int(val)].append(idx)

            # Convert to a regular list of lists if you want
            batch_idx_lists = list(batchable_indices.values())
            #timestep_tensor = timestep_tensor.cpu()
            # group by timestep
            for batch_idx_list in batch_idx_lists:
                #batch_idx_tensor = torch.tensor(batch_idx_list).to("cuda")
                #print(batch_idx_list)
                # print("noisepred ",noise_pred.device)
                # print("latents ",latents.device)
                batch_noise_preds = noise_pred[batch_idx_list]
                batch_latents = latents[batch_idx_list]
                # print("batch_noise_preds ",batch_noise_preds.device)
                # print("batch_latents ",batch_latents.device)
                # print("timestep ",int(timestep_tensor[batch_idx_list[0]]))
                #print(timestep_tensor.device)
                
                #print(batch_idx_tensor.device)
                #added eta=1.0 for noise reintroduction so identical latents can still have diversity
                new_latents = self.scheduler.step(batch_noise_preds, int(timestep_tensor[batch_idx_list[0]]), batch_latents, eta=1.0).prev_sample
                #new_latents = scheduler_output
                #print("new_latents ",scheduler_output.prev_sample.shape)
                latents[batch_idx_list] = new_latents

        return latents

#referenced from https://github.com/huggingface/diffusers/blob/b71269675ec1b85193107a691dd35c308e46f0a5/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L332
#for diffusers 0.35.2 and StableDiffusionPipeline
    def encode_prompt(
            self,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            clip_skip: Optional[int] = None,
        ):
            r"""
            Encodes the prompt into text encoder hidden states.

            Args:
                prompt (`str` or `List[str]`, *optional*):
                    prompt to be encoded
                device: (`torch.device`):
                    torch device
                num_images_per_prompt (`int`):
                    number of images that should be generated per prompt
                do_classifier_free_guidance (`bool`):
                    whether to use classifier free guidance or not
                negative_prompt (`str` or `List[str]`, *optional*):
                    The prompt or prompts not to guide the image generation. If not defined, one has to pass
                    `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                    less than `1`).
                prompt_embeds (`torch.Tensor`, *optional*):
                    Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                    provided, text embeddings will be generated from `prompt` input argument.
                negative_prompt_embeds (`torch.Tensor`, *optional*):
                    Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                    weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                    argument.
                lora_scale (`float`, *optional*):
                    A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
                clip_skip (`int`, *optional*):
                    Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                    the output of the pre-final layer will be used for computing the prompt embeddings.
            """
            # set lora scale so that monkey patched LoRA
            # function of text encoder can correctly access it
           

            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            if prompt_embeds is None:
                # textual inversion: process multi-vector tokens if necessary
            

                text_inputs = self.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids
                untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

                if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = self.tokenizer.batch_decode(
                        untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                    )
                    logger.warning(
                        "The following part of your input was truncated because CLIP can only handle sequences up to"
                        f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                    )

                if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                    attention_mask = text_inputs.attention_mask.to(device)
                else:
                    attention_mask = None

                if clip_skip is None:
                    prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                    prompt_embeds = prompt_embeds[0]
                else:
                    prompt_embeds = self.text_encoder(
                        text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                    )
                    # Access the `hidden_states` first, that contains a tuple of
                    # all the hidden states from the encoder layers. Then index into
                    # the tuple to access the hidden states from the desired layer.
                    prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                    # We also need to apply the final LayerNorm here to not mess with the
                    # representations. The `last_hidden_states` that we typically use for
                    # obtaining the final prompt representations passes through the LayerNorm
                    # layer.
                    prompt_embeds = self.text_encoder.text_model.final_layer_norm(prompt_embeds)

            if self.text_encoder is not None:
                prompt_embeds_dtype = self.text_encoder.dtype
            elif self.unet is not None:
                prompt_embeds_dtype = self.unet.dtype
            else:
                prompt_embeds_dtype = prompt_embeds.dtype

            prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

            # get unconditional embeddings for classifier free guidance
            if do_classifier_free_guidance and negative_prompt_embeds is None:
                uncond_tokens: List[str]
                if negative_prompt is None:
                    uncond_tokens = [""] * batch_size
                elif prompt is not None and type(prompt) is not type(negative_prompt):
                    raise TypeError(
                        f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                        f" {type(prompt)}."
                    )
                elif isinstance(negative_prompt, str):
                    uncond_tokens = [negative_prompt]
                elif batch_size != len(negative_prompt):
                    raise ValueError(
                        f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                        f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                        " the batch size of `prompt`."
                    )
                else:
                    uncond_tokens = negative_prompt

                # textual inversion: process multi-vector tokens if necessary

                max_length = prompt_embeds.shape[1]
                uncond_input = self.tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                    attention_mask = uncond_input.attention_mask.to(device)
                else:
                    attention_mask = None

                negative_prompt_embeds = self.text_encoder(
                    uncond_input.input_ids.to(device),
                    attention_mask=attention_mask,
                )
                negative_prompt_embeds = negative_prompt_embeds[0]

            if do_classifier_free_guidance:
                # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                seq_len = negative_prompt_embeds.shape[1]

                negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

                negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
                negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)


            return prompt_embeds, negative_prompt_embeds