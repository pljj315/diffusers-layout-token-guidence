import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
os.environ["ACCELERATE_LOG_LEVEL"] = "WARNING"
os.environ['HF_HOME'] = '/mnt/nfs/file_server/public/xy/huggingface'
from pipeline.layout_guidance_sdxl_inpaint_pipeline import StableDiffusionXLInpaintPipelineModify
from diffusers import UniPCMultistepScheduler
import torch
from PIL import ImageOps, Image
from diffusers.utils.pil_utils import make_image_grid
import numpy as np
import random

device = torch.device('cuda')
weight_dtype = torch.float16

pretrained_model_name_or_path = "/mnt/nfs/file_server/public/qirui/ckpt/SDXL_inpainting"  # 9通道sdxl inpaint模型
pipeline_inpaint = StableDiffusionXLInpaintPipelineModify.from_pretrained(
    pretrained_model_name_or_path,
    torch_dtype=weight_dtype,
    local_files_only = True,
    variant = "fp16")
pipeline_inpaint.scheduler = UniPCMultistepScheduler.from_config(pipeline_inpaint.scheduler.config)
pipeline_inpaint = pipeline_inpaint.to(device)


resolution = 1024
ori_img = Image.open("./data/cap.png").convert("RGB")
ori_mask_img =  Image.open("./data/cap_mask.png").convert("L")
new_mask = ImageOps.invert(ori_mask_img) # 把inpaint用作outpaint 
ori_img.resize((resolution, resolution))
ori_mask_img.resize((resolution, resolution))

prompt = "A cap and a bird, next to a lake, sunset"
get_indices = pipeline_inpaint.get_indices(prompt)
print('====检查indice词====',get_indices)

# 得到 mask的位置，并以mask左边位置作为bbox:
y_indices, x_indices = np.where(np.array(ori_mask_img) > 127)
x_min, y_min, x_max, y_max = np.min(x_indices), np.min(y_indices),np.max(x_indices), np.max(y_indices)
bboxes = [[0, y_min/resolution, x_min/resolution, y_max/resolution]] # 0-1!!!

# 调参：
max_guidance_step_list=[5,10,20,30]
max_guidance_iter_per_step_list = [1,2,5,10,20,30]

output_dir = './ouput/sdxl_inpaint/'
os.makedirs(output_dir, exist_ok=True)

num_validation_images = 2
seed = random.randint(0, 9999999)
generator = torch.Generator(device=device).manual_seed(seed)
for max_guidance_step in max_guidance_step_list:
    for max_guidance_iter_per_step in max_guidance_iter_per_step_list:  
        gen_images =[]
        for _ in range(num_validation_images): # 本pipeline的逻辑没有考虑到num_images_per_prompt！=1的情况
            image = pipeline_inpaint(
                prompt = prompt,
                image= ori_img, 
                mask_image = new_mask,

                token_indices=[[5]], # bird
                bboxes=bboxes,
                max_guidance_step= max_guidance_step,
                max_guidance_iter_per_step= max_guidance_iter_per_step,
                attn_res = (32,32),  # 32 或者 64 如果attn_res是None， 存储全部大小的attn

                height = resolution,
                width = resolution,
                num_inference_steps=30, 
                strength = 1.0,  
                generator=generator, 
                ).images[0]
            image = pipeline_inpaint.draw_box(image, bboxes)
            gen_images.append(image)
        grid = make_image_grid(gen_images, rows=1, cols=len(gen_images))
        grid.save(output_dir + f'/step_{max_guidance_step}_iter_{max_guidance_iter_per_step}_{seed}.jpg')
# latent更新：
# 调参max_guidance_step_list/max_guidance_iter_per_step_list 对效果会有影响