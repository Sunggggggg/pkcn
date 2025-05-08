import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import math
from einops import rearrange
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp, Attention

def sinusoidal_positional_encoding(seq_len, embed_dim):
    position = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
    pe = torch.zeros(seq_len, embed_dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.reshape(1, seq_len, 1, embed_dim)  # (1, seq_len, 1, embed_dim)
    return pe

class ReltiveAttention(nn.Module):
    def __init__(self, dim, num_view, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position = nn.Parameter(torch.zeros(num_heads, num_view, num_view))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x : [BJ, N, D]
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))        # [BJ, N, N]

        relative_position_bias = self.relative_position.unsqueeze(0)
        attn = attn + relative_position_bias

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, num_view=None):
        super().__init__()

        self.norm1 = norm_layer(dim)
        if num_view is not None :
            self.attn = ReltiveAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, num_view=num_view)
        else :
            self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class SVTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_joint,
                 num_heads=8, drop_rate=0., drop_path_rate=0., mlp_ratio=2.,
                 qkv_bias=True, qk_scale=None, attn_drop_rate=0., depth=3,):
        super(SVTransformer, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, num_joint, embed_dim)) 
        self.register_buffer("view_encoding", sinusoidal_positional_encoding(1000, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.depth = depth
        norm_layer = nn.LayerNorm

        self.SpatialBlocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.ViewBlock = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm_s = norm_layer(embed_dim)
        self.norm_v = norm_layer(embed_dim)
        
    def forward(self, x):
        """ x : [B, N, J, C]
        """
        B, N, J, C = x.size()
        x = self.embedding(x)
        x = x + self.pos_encoding[:, :, :J, :] + self.view_encoding[:, :N, :, :]
        
        for i in range(self.depth) :
            blk1 = self.SpatialBlocks[i]
            blk2 = self.ViewBlock[i]
            
            x = rearrange(x, 'b n j c -> (b n) j c')
            x = blk1(x)
            x = self.norm_s(x)
            
            x = rearrange(x, '(b n) j c -> (b j) n c', b=B, n=N)
            x = blk2(x)
            x = self.norm_v(x)
        
            x = rearrange(x, '(b j) n c -> b n j c', b=B, j=J)
        
        x = x.mean(dim=1)   # [B, J, C]
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, hidden_dim, num_layers):
        super(TransformerEncoder, self).__init__()

        # Embedding layer
        self.embedding = nn.Linear(input_dim, embed_dim)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 1000, embed_dim))

        # Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src):
        src = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        # Encoding
        memory = self.encoder(src)
        return memory

class TransformerDecoder(nn.Module):
    def __init__(self, output_dim, embed_dim, num_heads, hidden_dim, num_layers):
        super(TransformerDecoder, self).__init__()

        # Embedding layer
        self.embedding = nn.Linear(output_dim, embed_dim)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 1000, embed_dim))

        # Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection
        self.output_layer = nn.Linear(embed_dim, output_dim)

    def forward(self, tgt, memory):
        # Embedding with positional encoding
        tgt = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
        # Decoding
        output = self.decoder(tgt, memory)
        # Final output projection
        return self.output_layer(output)

class SVRTransformer(nn.Module):
    """
    Spatial, View-aware, Ray direciton
    """
    def __init__(self, input_dim, embed_dim, num_joint,
                 num_heads=8, drop_rate=0., drop_path_rate=0., mlp_ratio=2.,
                 qkv_bias=True, qk_scale=None, attn_drop_rate=0., depth=3,):
        super(SVRTransformer, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Linear(input_dim, embed_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, num_joint, embed_dim)) 
        self.ray_encoding = nn.Linear(embed_dim+3, embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.depth = depth
        norm_layer = nn.LayerNorm

        self.SpatialBlocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.ViewBlock = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm_s = norm_layer(embed_dim)
        self.norm_v = norm_layer(embed_dim)
        
    def forward(self, x, ray):
        """ 
        B : 사람 수
        x   : [B, N, J, C]
        ray : [B, N, 3]
        """
        B, N, J, C = x.size()
        x = self.embedding(x)
        x = x + self.pos_encoding[:, :, :J, :]              # [B, N, J, C]
        ray_expand = ray.unsqueeze(-2).repeat(1, 1, J, 1) 
        x = torch.cat([x, ray_expand], dim=-1)              # [B, N, J, C+3]
        x = self.ray_encoding(x)                            # [B, N, J, C]
        
        for i in range(self.depth) :
            blk1 = self.SpatialBlocks[i]
            blk2 = self.ViewBlock[i]
            
            x = rearrange(x, 'b n j c -> (b n) j c')
            x = blk1(x)
            x = self.norm_s(x)
            
            x = rearrange(x, '(b n) j c -> (b j) n c', b=B, n=N)
            x = blk2(x)
            x = self.norm_v(x)
        
            x = rearrange(x, '(b j) n c -> b n j c', b=B, j=J)
        
        x = x.mean(dim=1)   # [B, J, C]
        return x

class Rel_SVTransformer(nn.Module):
    """
    Spatial, View-aware, Relative positional encoding
    """
    def __init__(self, input_dim, embed_dim, num_joint, num_view=4,
                 num_heads=8, drop_rate=0., drop_path_rate=0., mlp_ratio=2.,
                 qkv_bias=True, qk_scale=None, attn_drop_rate=0., depth=3,):
        super(Rel_SVTransformer, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Linear(input_dim, embed_dim)

        # Positional encoding
        self.spatio_pos_encoding = nn.Parameter(torch.randn(1, 1, num_joint, embed_dim)) 
        self.relative_pos_encoding = nn.Parameter(torch.zeros(num_view, num_view)) 
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.depth = depth
        norm_layer = nn.LayerNorm

        self.SpatialBlocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.ViewBlock = nn.ModuleList(
            [Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, num_view=num_view)
            for i in range(depth)])

        self.norm_s = norm_layer(embed_dim)
        self.norm_v = norm_layer(embed_dim)
        
    def forward(self, x):
        """ 
        B : 사람 수
        x   : [B, N, J, C]
        """
        B, N, J, C = x.size()
        x = self.embedding(x)
        x = x + self.spatio_pos_encoding[:, :, :J, :]       # [B, N, J, C]
        
        for i in range(self.depth) :
            blk1 = self.SpatialBlocks[i]
            blk2 = self.ViewBlock[i]
            
            x = rearrange(x, 'b n j c -> (b n) j c')
            x = blk1(x)
            x = self.norm_s(x)
            
            x = rearrange(x, '(b n) j c -> (b j) n c', b=B, n=N)
            x = blk2(x)
            x = self.norm_v(x)
        
            x = rearrange(x, '(b j) n c -> b n j c', b=B, j=J)
        
        x = x.mean(dim=1)   # [B, J, C]
        return x

class RoPE_SVTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_joint,
                 num_heads=8, drop_rate=0., drop_path_rate=0., mlp_ratio=2.,
                 qkv_bias=True, qk_scale=None, attn_drop_rate=0., depth=3,):
        super(RoPE_SVTransformer, self).__init__()
        from .rope import RoPE
        
        # Embedding layer
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, num_joint, embed_dim)) 
        self.view_pos_encoding = RoPE(embed_dim, max_seq_len=8)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.depth = depth
        norm_layer = nn.LayerNorm

        self.SpatialBlocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.ViewBlock = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm_s = norm_layer(embed_dim)
        self.norm_v = norm_layer(embed_dim)
        
    def forward(self, x):
        """ x : [B, N, J, C]
        """
        B, N, J, C = x.size()
        x = self.embedding(x)
        x = x + self.pos_encoding[:, :, :J, :]  # [B, N, J, C]
        x = rearrange(x, 'b n j c -> (b j) n c')
        x = self.view_pos_encoding.rotate_embedding(x)
        x = rearrange(x, '(b j) n c -> b n j c', b=B, j=J)
        
        for i in range(self.depth) :
            blk1 = self.SpatialBlocks[i]
            blk2 = self.ViewBlock[i]
            
            x = rearrange(x, 'b n j c -> (b n) j c')
            x = blk1(x)
            x = self.norm_s(x)
            
            x = rearrange(x, '(b n) j c -> (b j) n c', b=B, n=N)
            x = blk2(x)
            x = self.norm_v(x)
        
            x = rearrange(x, '(b j) n c -> b n j c', b=B, j=J)
        
        x = x.mean(dim=1)   # [B, J, C]
        return x
    
if __name__ == "__main__":
    # model = SVTransformer(input_dim=512, embed_dim=256, num_joint=24)
    model = Rel_SVTransformer(input_dim=512, embed_dim=256, num_joint=24, num_view=4)

    src = torch.randn(1, 4, 24, 512)   # (Batch, Seq, Feature)
    ray = torch.randn(1, 4, 3)         # (Batch, Seq, Feature)

    output = model(src)
    print(output.shape)