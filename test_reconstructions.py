import os
import PIL
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np
import yaml
import torch
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel, GumbelVQ

import torchvision.transforms as T
import torchvision.transforms.functional as TF

font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf", 22)

def preprocess(img, target_image_size=256):
    s = min(img.size)
    
    if s < target_image_size:
        raise ValueError(f'min dim for image {s} < {target_image_size}')
        
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2*[target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)
    return img

def stack_reconstructions(input, x0, x1, x2, x3, x4, x5, titles=[]):
  assert input.size == x1.size
  w, h = input.size[0], input.size[1]
  img = Image.new("RGB", (7*w, h))
  img.paste(input, (0, 0))
  img.paste(x0, (1*w, 0))
  img.paste(x1, (2*w, 0))
  img.paste(x2, (3*w, 0))
  img.paste(x3, (4*w, 0))
  img.paste(x4, (5*w, 0))
  img.paste(x5, (6*w, 0))

  for i, title in enumerate(titles):
    ImageDraw.Draw(img).text((i*w, 0), f'{title}', (255, 255, 255), font=font)  # coordinates, text, color, font
  img.save("input_reconstructions.jpg")

def load_config(config_path, display=False):
  config = OmegaConf.load(config_path)
  if display:
    print(yaml.dump(OmegaConf.to_container(config)))
  return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
  if is_gumbel:
    model = GumbelVQ(**config.model.params)
  else:
    model = VQModel(**config.model.params)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
  return model.eval()

def preprocess_vqgan(x):
  x = 2.*x - 1.
  return x

def custom_to_pil(x):
  x = x.detach().cpu()
  x = torch.clamp(x, -1., 1.)
  x = (x + 1.)/2.
  x = x.permute(1, 2, 0).numpy()
  x = (255 * x).astype(np.uint8)
  x = Image.fromarray(x)
  if not x.mode == "RGB":
    x = x.convert("RGB")
  return x

def reconstruct_with_vqgan(x, model):
  # could also use model(x) for reconstruction but use explicit encoding and decoding here
  z, _, [_, _, indices] = model.encode(x)
  print(f"VQGAN --- {model.__class__.__name__}: latent shape: {z.shape[2:]}")
  print("z.max(), z.min():", z.max(), z.min())
  print("indices.shape, indices.max(), indices.min():", indices.shape, indices.max(), indices.min())
  xrec = model.decode(z)
  return xrec

def reconstruction_pipeline(url, size=256):
  x_vqgan = preprocess(PIL.Image.open(url), target_image_size=size)
  x_vqgan = x_vqgan.to("cuda:0")
  
  print(f"input is of size: {x_vqgan.shape}")
  x0 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), model_openimages)
  x1 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), model_imagenet)
  x2 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), model_say)
  x3 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), model_s)
  x4 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), model_a)
  x5 = reconstruct_with_vqgan(preprocess_vqgan(x_vqgan), model_y)

  img = stack_reconstructions(custom_to_pil(preprocess_vqgan(x_vqgan[0])), 
                              custom_to_pil(x0[0]), custom_to_pil(x1[0]), custom_to_pil(x2[0]), 
                              custom_to_pil(x3[0]), custom_to_pil(x4[0]), custom_to_pil(x5[0]),
                              titles=titles
                              )

  return img

if __name__ == "__main__":
    config_imagenet = load_config("logs/vqgan_imagenet_f16_16384/configs/model.yaml", display=False)
    model_imagenet = load_vqgan(config_imagenet, ckpt_path="logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt").to("cuda:0")

    config_openimages = load_config("logs/vqgan_gumbel_f8/configs/model.yaml", display=False)
    model_openimages = load_vqgan(config_openimages, ckpt_path="logs/vqgan_gumbel_f8/checkpoints/last.ckpt", is_gumbel=True).to("cuda:0")

    config_say = load_config("logs/2022-08-07T20-31-31_custom_vqgan_32x32_say/configs/2022-08-09T20-37-58-project.yaml", display=False)
    model_say = load_vqgan(config_say, ckpt_path="logs/2022-08-07T20-31-31_custom_vqgan_32x32_say/checkpoints/last.ckpt").to("cuda:0")

    config_s = load_config("logs/2022-08-15T21-05-50_custom_vqgan_32x32_s/configs/2022-08-19T19-37-48-project.yaml", display=False)
    model_s = load_vqgan(config_s, ckpt_path="logs/2022-08-15T21-05-50_custom_vqgan_32x32_s/checkpoints/last.ckpt").to("cuda:0")

    config_a = load_config("logs/2022-08-17T19-36-46_custom_vqgan_32x32_a/configs/2022-08-20T21-36-31-project.yaml", display=False)
    model_a = load_vqgan(config_a, ckpt_path="logs/2022-08-17T19-36-46_custom_vqgan_32x32_a/checkpoints/last.ckpt").to("cuda:0")

    config_y = load_config("logs/2022-08-18T21-35-31_custom_vqgan_32x32_y/configs/2022-08-21T19-38-06-project.yaml", display=False)
    model_y = load_vqgan(config_y, ckpt_path="logs/2022-08-18T21-35-31_custom_vqgan_32x32_y/checkpoints/last.ckpt").to("cuda:0")

    titles=["image", "openimages_32x32_8192", "imagenet_16x16_16384", "say_32x32_8192", "s_32x32_8192", "a_32x32_8192", "y_32x32_8192"]
    test_img_dir = '/misc/vlgscratch4/LakeGroup/emin/taming-transformers/test_images'
    test_img = 'saycam_5.jpeg' 
  
    reconstruction_pipeline(url=os.path.join(test_img_dir, test_img), size=256)
