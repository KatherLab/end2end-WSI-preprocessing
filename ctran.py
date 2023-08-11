try:
    from timm.layers.helpers import to_2tuple
except ModuleNotFoundError:
    from timm.models.layers.helpers import to_2tuple
import timm
import torch.nn as nn
import torch

class ConvStem(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        

        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

def swin(pretrained=False):                       # 48.5M params                    
    model = timm.create_model('swin_tiny_patch4_window7_224', embed_layer=ConvStem, pretrained=False)
    if pretrained:
        checkpoint_path = "ctranspath.pth"
        ctranspath = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(ctranspath['model'], strict=False)
    return model

def tiny_vit():                                        # 
    model = timm.create_model('vit_tiny_patch16_224', embed_layer=ConvStem, pretrained=False) 
    return model

def pvt_v2_b2_li():
    model = timm.create_model('pvt_v2_b2_li', pretrained=False) # 22.6M params
    return model

def pvt_v2_b2():
    model = timm.create_model('pvt_v2_b2', pretrained=False) # 25.4M params
    return model

def convnextv2():
    model = timm.create_model('convnextv2_nano', pretrained=False) # 28.6M params
    return model

def maxvit_small():
    model = timm.create_model("maxvit_rmlp_small_rw_224",pretrained=False)
    return model

def maxvit_tiny():
    model = timm.create_model("maxvit_tiny_rw_224",pretrained=False)
    return model