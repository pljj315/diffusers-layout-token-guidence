基于diffusers实现， 对sdxl模型推理过程使用cross_attn的map去计算loss，并根据loss对latent进行更新；其中loss的设计至关重要，可以实现不同的目标，比如layout控制，token增强等
