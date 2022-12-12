import argparse
import logging
from dataclasses import dataclass
import os
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from diffusers import AutoencoderKL
from diffusers import UNet2DConditionModel, PNDMScheduler, LMSDiscreteScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
from typing import List
from huggingface_hub import login


class Pipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.vae = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="vae", use_auth_token=True,
        ).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="unet", use_auth_token=True,
        ).to(self.device)
        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )

    def _get_text_embeds(self, prompt):
        # Tokenize text and get embeddings
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(
            [""] * len(prompt),
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(self.device)
            )[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    def _encode_img_latents(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]

        img_arr = np.stack([np.array(img) for img in imgs], axis=0)
        img_arr = img_arr / 255.0
        img_arr = torch.from_numpy(img_arr).float().permute(0, 3, 1, 2)
        img_arr = 2 * (img_arr - 0.5)

        latent_dists = self.vae.encode(img_arr.to(self.device))
        latent_samples = latent_dists["latent_dist"].sample()
        latent_samples *= 0.18215

        return latent_samples

    def _decode_img_latents(self, latents):
        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents)["sample"]

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in imgs]
        return pil_images

    def _produce_latents(
        self,
        text_embeddings,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
        return_all_latents=False,
        start_step=10,
    ):
        if latents is None:
            latents = torch.randn(
                (
                    text_embeddings.shape[0] // 2,
                    self.unet.in_channels,
                    height // 8,
                    width // 8,
                )
            )

        latents = latents.to(self.device)

        self.scheduler.set_timesteps(num_inference_steps)
        if start_step > 0:
            start_timestep = self.scheduler.timesteps[start_step]
            start_timesteps = start_timestep.repeat(latents.shape[0]).long()

            noise = torch.randn_like(latents)
            latents = self.scheduler.add_noise(latents, noise, start_timesteps)

        latent_hist = [latents]
        # with torch.autocast(self.device):
        for t in tqdm(self.scheduler.timesteps[start_step:]):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                )["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]
            latent_hist.append(latents)

        if not return_all_latents:
            return latents

        all_latents = torch.cat(latent_hist, dim=0)
        return all_latents

    def encode_img(self, image):
        return self._encode_img_latents([image])

    def prompt_to_img(
        self,
        prompts,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        latents=None,
        return_all_latents=False,
        batch_size=2,
        start_step=0,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        # Prompts -> text embeds
        text_embeds = self._get_text_embeds(prompts)

        # Text embeds -> img latents
        latents = self._produce_latents(
            text_embeds,
            height=height,
            width=width,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            return_all_latents=return_all_latents,
            start_step=start_step,
        )

        # Img latents -> imgs
        all_imgs = []
        for i in tqdm(range(0, len(latents), batch_size)):
            imgs = self._decode_img_latents(latents[i : i + batch_size])
            all_imgs.extend(imgs)

        return all_imgs


class OptionImage:
    def get(self, encode):
        pass

    @property
    def start_step(self):
        pass

    @property
    def num_inference_steps(self):
        pass

    @property
    def width(self):
        pass

    @property
    def height(self):
        pass


class RandomImage(OptionImage):
    def get(self, _):
        return None

    @property
    def start_step(self):
        return 0

    @property
    def num_inference_steps(self):
        return 20

    @property
    def width(self):
        return 512

    @property
    def height(self):
        return 512


class PathImage(OptionImage):
    def __init__(self, path: str):
        super().__init__()
        self._path = path
        self._img = None

    @staticmethod
    def expand2square(pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def get(self, encode):
        img = Image.open(self._path)
        img.thumbnail((512, 512), Image.Resampling.LANCZOS)

        self._img = img

        # img = PathImage.expand2square(img, (0, 0, 0))

        return encode(img)

    @property
    def start_step(self):
        return 20

    @property
    def num_inference_steps(self):
        return 30

    @property
    def width(self):
        return self._img.width

    @property
    def height(self):
        return self._img.height


class CameraImage(OptionImage):
    def __init__(self):
        super().__init__()
        self._img = None

    def get(self, encode):
        vidcap = cv2.VideoCapture(0)

        if vidcap.isOpened():
            while(True):
                _, frame = vidcap.read()

                cv2.imshow("Frame",frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            logging.error("Cannot open camera")

        vidcap.release()
        cv2.destroyAllWindows()

        img = Image.fromarray(frame)
        img.thumbnail((512, 512), Image.Resampling.LANCZOS)

        self._img = img

        return encode(img)

    @property
    def start_step(self):
        return 20

    @property
    def num_inference_steps(self):
        return 30

    @property
    def width(self):
        return self._img.width

    @property
    def height(self):
        return self._img.height


@dataclass
class Options:
    prompt: str
    image: OptionImage
    output: str


def _info(opt: Options) -> None:
    logging.info(f"Using prompt {opt.prompt} for image {opt.image}.")


def main(opt: Options):
    _info(opt)
    pipeline = Pipeline()

    prompt = opt.prompt
    latents = opt.image.get(pipeline.encode_img)
    start_step = opt.image.start_step
    num_inference_steps = opt.image.num_inference_steps
    height = opt.image.height
    width = opt.image.width

    image = pipeline.prompt_to_img(
        prompt,
        latents=latents,
        height=height,
        width=width,
        start_step=start_step,
        num_inference_steps=num_inference_steps,
    )[0]
    image.save(f"{opt.output}/{prompt}.png")

    _, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    plt.title(prompt)
    plt.axis("off")
    plt.show()


def get_options() -> Options:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt",
        dest="prompt",
        type=str,
        default="example",
        help="Prompt to use with the AI",
    )
    parser.add_argument(
        "--source",
        dest="source",
        type=str,
        default=None,
        help="Image to use as starting point",
    )
    parser.add_argument(
        "--camera",
        dest="camera",
        action="store_true",
        default=False,
        help="Use webcam to capture an image instead of source",
    )
    parser.add_argument(
        "--output",
        dest="output",
        type=str,
        default="data",
        help="Output directory to save the results to",
    )

    args = parser.parse_args()

    source = args.source
    camera = args.camera

    if source is not None:
        image = PathImage(source)
    elif camera:
        image = CameraImage()
    else:
        image = RandomImage()

    output = args.output
    os.makedirs(output, exist_ok=True)

    return Options(prompt=args.prompt, image=image, output=output)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
    login(huggingface_token)

    logging.basicConfig(
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.getenv("LOG_FILE", "app.log")),
        ],
        level=logging.DEBUG,
        format="%(levelname)s: %(asctime)s \
            pid:%(process)s module:%(module)s %(message)s",
        datefmt="%d/%m/%y %H:%M:%S",
    )

    main(get_options())
