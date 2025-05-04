import torch
from torch import nn

model_ver_list = ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14']    # [384, 768, 1024]

IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]

class Dinov2Backbone(nn.Module):
    def __init__(self, name='dinov2_vits14', pretrained=True):
        super().__init__()
        self.name = name
        self.encoder = torch.hub.load('facebookresearch/dinov2', self.name, pretrained=pretrained)
        self.patch_size = self.encoder.patch_size
        self.embed_dim = self.encoder.embed_dim
        self.backbone_channels = self.encoder.embed_dim

    def forward(self, x, norm=True):
        """
        Encode a RGB image using a ViT-backbone
        Args:
            - x: torch.Tensor of shape [bs,3,w,h]
        Return:
            - y: torch.Tensor of shape [bs,k,d] - image in patchified mode
        """
        B, H, W, C = x.shape
        h, w = H // self.patch_size, W // self.patch_size
        
        if norm :
            mean = torch.tensor(IMG_NORM_MEAN, dtype=x.dtype, device=x.device).reshape(1, 1, 1, 3)
            std = torch.tensor(IMG_NORM_STD, dtype=x.dtype, device=x.device).reshape(1, 1, 1, 3)
        
            x = x / 255.            # [B, H, W, 3]
            x = (x - mean) / std

        x = x.permute(0, 3, 1, 2)
        y = self.encoder.get_intermediate_layers(x)[0] # ViT-L+896x896: [bs,4096,1024] - [bs,nb_patches,emb]
        y = y.contiguous().reshape(B, h, w, -1).permute(0, 3, 1, 2)

        return y

    def load_pretrain_params(self):
        pass
    
if __name__ == '__main__':
    x = torch.rand((1, 3, 448, 448))
    for ver in model_ver_list :
        model = Dinov2Backbone(name=ver, pretrained=True)
        y = model(x)
        print(y.shape)