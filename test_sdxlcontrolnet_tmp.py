import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
os.environ["ACCELERATE_LOG_LEVEL"] = "WARNING"
os.environ['HF_HOME'] = '/mnt/nfs/file_server/public/xy/huggingface'
from pipeline.layout_guidance_sdxlcontrolnet_pipeline_tmp import StableDiffusionXLControlNetPipelineModify
from diffusers import UniPCMultistepScheduler, AutoencoderKL, ControlNetModel
import torch
from PIL import ImageOps, Image
from diffusers.utils.pil_utils import make_image_grid
import numpy as np
import random
import cv2
from torchvision import transforms

def get_cond_color(ori_image, cond_img=None, rand=False, rotate_method = None, random_threshold = 0):
    np_img = np.asarray(ori_image)
    H, W, c = np_img.shape
    np_cond_img = np.asarray(cond_img)
    np_cond_img = cv2.resize(np_cond_img, (W,H))
    try:
        if rotate_method != None :
            image_mask_dict = rotate_method(image=ori_image, mask = cond_img)
            ori_image = image_mask_dict['image']
            cond_img = image_mask_dict['mask']

        if rand and random.random()> random_threshold:
            np_img = np.asarray(ori_image)
            H, W, c = np_img.shape
            np_cond_img = np.asarray(cond_img)
            np_cond_img = cv2.resize(np_cond_img, (W,H))

            mask = np_cond_img[:,:,0]
            h,w= np.where(mask>127)
            min_h,min_w,max_h,max_w = np.min(h),np.min(w),np.max(h),np.max(w)
            
            obj_w = max_w - min_w
            obj_h = max_h - min_h

            
            if obj_w>obj_h:
                crop_top_w = random.randint(0,min_w)
                if crop_top_w+256>W:
                    crop_top_w = W-256
                    crop_down_w = W
                else:
                    crop_down_w = random.randint(max(max_w,crop_top_w+256), W)
                len_w = crop_down_w - crop_top_w

                crop_top_h = random.randint(max(max_h-len_w,0),min_h)
                if crop_top_h+len_w>H:
                    crop_top_h = max(H-len_w,0)
                    crop_down_h = H
                else:
                    crop_down_h = crop_top_h+len_w
                np_img = np_img[crop_top_h:crop_down_h,crop_top_w:crop_down_w]
                np_cond_img = np_cond_img[crop_top_h:crop_down_h,crop_top_w:crop_down_w]
                left_top_h,felt_top_w = max(crop_top_h,0),max(crop_top_w,0)
            else:
                crop_top_h = random.randint(0,min_h)
                if crop_top_h+256>H: # 256 可能小
                    crop_top_h = H-256
                    crop_down_h = H
                else:
                    crop_down_h = random.randint(max(max_h,crop_top_h+256), H)
                len_h = crop_down_h - crop_top_h
                crop_top_w = random.randint(max(max_w-len_h,0),min_w)
                if crop_top_w+len_h>W:
                    crop_top_w = max(W-len_h,0)
                    crop_down_w = W
                else:
                    crop_down_w = crop_top_w+len_h
                np_img = np_img[crop_top_h:crop_down_h,crop_top_w:crop_down_w]
                np_cond_img = np_cond_img[crop_top_h:crop_down_h,crop_top_w:crop_down_w]
                left_top_h,felt_top_w = max(crop_top_h, 0),max(crop_top_w, 0)
        else:
            left_top_h,felt_top_w = 0,0
    except:
        left_top_h,felt_top_w = 0,0
    cond = np.zeros_like(np_img)
    cond[np_cond_img>127] = np_img[np_cond_img>127]
    cond = np.concatenate([cond, np_cond_img[:,:,-1:]],-1)
    cond = Image.fromarray(cond)
    ori_img = Image.fromarray(np_img)
    out_mask = Image.fromarray(np_cond_img)
    return cond, ori_img, out_mask, left_top_h, felt_top_w


device = torch.device('cuda')
weight_dtype = torch.float16

vae_path = 'madebyollin/sdxl-vae-fp16-fix'
vae = AutoencoderKL.from_pretrained(vae_path , 
                                    force_upcast=False,
                                    torch_dtype=weight_dtype,
                                    local_files_only = True)
vae.requires_grad_(False)

controlnet_model_name_or_path = '/mnt/nfs/file_server/public/xy/data/laion_aesthetic_6/train_outpainting_768ft_aug_02_proweb/pipeline-1600'
controlnet = ControlNetModel.from_pretrained(
    controlnet_model_name_or_path,
    torch_dtype=weight_dtype,
    local_files_only = True,)

pretrained_sdxl_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
pipeline_controlnet = StableDiffusionXLControlNetPipelineModify.from_pretrained(
    pretrained_sdxl_model_name_or_path,
    vae = vae,
    controlnet=controlnet,
    torch_dtype=weight_dtype,
    variant='fp16',
    local_files_only = True)
pipeline_controlnet.scheduler = UniPCMultistepScheduler.from_config(pipeline_controlnet.scheduler.config)
pipeline_controlnet = pipeline_controlnet.to(device)

resolution=1024
conditioning_image_transforms = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),  ])

prompt = "A cap and a bird, next to a lake, sunset"
get_indices = pipeline_controlnet.get_indices(prompt)
print('====检查indice词====',get_indices)

ori_img = Image.open("./data/cap.png").convert("RGB")
ori_mask_img =  Image.open("./data/cap_mask.png").convert("L")
ori_img.resize((resolution, resolution))
ori_mask_img.resize((resolution, resolution))
aug_validation_image, aug_ori_img, out_mask, left_top_h, left_top_w = get_cond_color(ori_img, cond_img=ori_mask_img.convert('RGB'), rand=False ) # seg 与 mask 合并为四通道(4,H,W) 作为输入
tensor_validation_image = conditioning_image_transforms(aug_validation_image)
mask = tensor_validation_image[3:,:,:]       
mask = torch.cat([mask, mask, mask, mask])
tensor_validation_image[mask==0] = -1        
tensor_validation_image = torch.unsqueeze(tensor_validation_image, 0)

# 得到 mask的位置，并以mask左边位置作为bbox:
y_indices, x_indices = np.where(np.array(ori_mask_img) > 127)
x_min, y_min, x_max, y_max = np.min(x_indices), np.min(y_indices),np.max(x_indices), np.max(y_indices)
bboxes = [[0, y_min/resolution, x_min/resolution, y_max/resolution]] # 0-1!!!

# 参数设置：
max_guidance_step_list=[5,10,20,30]
max_guidance_iter_per_step_list = [1,2,5,10,20,30]

output_dir = './ouput/sdxlcontrolnet_tmp/'
os.makedirs(output_dir, exist_ok=True)

num_validation_images = 1
# seed = random.randint(0, 9999999)
seed = 5998283
generator = torch.Generator(device=device).manual_seed(seed)
for max_guidance_step in max_guidance_step_list:
    for max_guidance_iter_per_step in max_guidance_iter_per_step_list:
        gen_images =[]
        for i in range(num_validation_images): # 本pipeline的逻辑没有考虑到num_images_per_prompt！=1的情况
            image = pipeline_controlnet(
                prompt = prompt,
                image= tensor_validation_image, 
                
                token_indices=[[5]],# bird
                bboxes=bboxes,
                max_guidance_step= max_guidance_step,
                max_guidance_iter_per_step= max_guidance_iter_per_step,
                attn_res = (32,32),  # 32 或者 64 如果attn_res是None， 存储全部大小的attn

                height = resolution,
                width = resolution,
                num_inference_steps=30, 
                generator=generator, 
                num_images_per_prompt =2,
                ).images[0]
            image = pipeline_controlnet.draw_box(image, bboxes)
            gen_images.append(image)
        grid = make_image_grid(gen_images, rows=1, cols=len(gen_images))
        grid.save(output_dir + f'/step_{max_guidance_step}_iter_{max_guidance_iter_per_step}_{seed}.jpg')
        
# latent更新：
# 调参max_guidance_step_list/max_guidance_iter_per_step_list 对效果会有影响