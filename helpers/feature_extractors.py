import hashlib
from pathlib import Path
from marugoto.marugoto.extract.xiyue_wang.RetCLL import ResNet
from marugoto.marugoto.extract.ctranspath.swin_transformer import swin_tiny_patch4_window7_224, ConvStem
import torch
import torch.nn as nn
import PIL
import numpy as np
import os
from vits import vit_conv_base
import timm
from ctran import swin

class FeatureExtractor:
    def __init__(self, model_type):
        self.model_type = model_type

    def init_feat_extractor(self, checkpoint_path: str, **kwargs):
        """Extracts features from slide tiles.
        Args:
            checkpoint_path:  Path to the model checkpoint file.
        """
        sha256 = hashlib.sha256()
        with open(checkpoint_path, 'rb') as f:
            while True:
                data = f.read(1 << 16)
                if not data:
                    break
                sha256.update(data)

        if self.model_type == 'retccl':
            assert sha256.hexdigest() == '931956f31d3f1a3f6047f3172b9e59ee3460d29f7c0c2bb219cbc8e9207795ff'

            model = ResNet.resnet50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
            pretext_model = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            model.fc = nn.Identity()
            model.load_state_dict(pretext_model, strict=True)

            if torch.cuda.is_available():
                model = model.cuda()

            print("RetCCL model successfully initialised...")
            model_name='xiyuewang-retcll-931956f3'
            return model, model_name
        
        if self.model_type == 'pvt':
            model = timm.create_model('pvt_v2_b2_li', pretrained=False) # 22.6M params # output states: 42M params
            model.head = nn.Identity()
            pvt = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            model.load_state_dict(pvt,strict=True)

            if torch.cuda.is_available():
                model = model.cuda()
                
            print(f"{self.model_type} model successfully initialised...")
                
            return model,self.model_type 
        
        if self.model_type == 'convnext':
            model = timm.create_model('convnextv2_tiny', pretrained=False) # 22.6M params # output states: 42M params
            model.head.fc = nn.Identity()
            pvt = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            model.load_state_dict(pvt,strict=True)

            if torch.cuda.is_available():
                model = model.cuda()
                
            print(f"{self.model_type} model successfully initialised...")
                
            return model,self.model_type 

        elif self.model_type == 'ctranspath':
            assert sha256.hexdigest() == '7c998680060c8743551a412583fac689db43cec07053b72dfec6dcd810113539'

            model = swin_tiny_patch4_window7_224(embed_layer=ConvStem, pretrained=False)
            model.head = nn.Identity()

            ctranspath = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            model.load_state_dict(ctranspath['model'], strict=True)
            
            if torch.cuda.is_available():
                model = model.cuda()

            print("CTransPath model successfully initialised...")
            model_name='xiyuewang-ctranspath-7c998680'

            return model, model_name
        
        if 'swin' in self.model_type:

            # model = swin_tiny_patch4_window7_224(embed_layer=ConvStem, pretrained=False)
            # model.head = nn.Identity()
        

            # state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            # model.load_state_dict(state_dict, strict=True)
            # if torch.cuda.is_available():
            #     model = model.cuda()

            model = swin()
            model.head = nn.Identity()
            swin_chkpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            model.load_state_dict(swin_chkpt,strict=True)
            
            if torch.cuda.is_available():
                model = model.cuda()
            
            print(f"{self.model_type} model successfully initialised...")

            return model, self.model_type
 
        if 'dinoV2' in self.model_type:
            
            #model = vit_base(patch_size=14)
            #dinoV2 = torch.load("/mnt/SATELLITE_03/tim_warmup/dino/dinov2_vitb14_pretrain.pth")
            
            #model.load_state_dict(dinoV2,strict=False)
            if "epoch" in self.model_type:
                model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
                state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
                model.load_state_dict(state_dict, strict=True)
            
            else:
                if len(self.model_type.split("-"))>1:
                    model_size=self.model_type.split("-")[-1].lower()
                    model = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_size}14')
                else:
                    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            
            if torch.cuda.is_available():
                model = model.cuda()
            
            print(f"{self.model_type} model successfully initialised...")
            return model, self.model_type

        if 'moco-vit' in self.model_type:
            
            model = vit_conv_base() 
            model.head = nn.Identity()
            
            state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict, strict=True)
            if torch.cuda.is_available():
                model = model.cuda()

            print(f"{self.model_type} model successfully initialised...")

            return model, self.model_type


        if self.model_type == 'resnet50':
		
            #model = ResNet.resnet50(num_classes=128, pretrained=True, mlp=False, two_branch=False, normlinear=True)
            #model.fc = nn.Identity()
            model = ResNet.resnet50(num_classes=1000, pretrained=True, mlp=False, two_branch=False, normlinear=True)
            model.fc = nn.Identity()

            if torch.cuda.is_available():
                model = model.cuda()

            print("ResNet50 model successfully initialised...")

            return model, self.model_type


        else:
            raise ValueError('Invalid model type')
        
        
        

# test extract_xiyuewang_features_ function
def test_extract_xiyuewang_features_():
    img = np.random.randint(0, 255, size=(1000, 1000, 3), dtype=np.uint8)
    feature_extractor = FeatureExtractor('retccl')
    feature_extractor.extract_features(norm_wsi_img=img, wsi_name='test', coords=[(0,0)], checkpoint_path='.', outdir='.')

    # test that the output file exists
    assert os.path.exists('test.h5')

    # test that the output file is not empty
    assert os.path.getsize('test.h5') > 0

    # remove the output file
    os.remove('test.h5')
