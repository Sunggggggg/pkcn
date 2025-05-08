import torch
import torch.nn as nn
from einops import repeat, rearrange
from torch.cuda.amp import autocast

def get_encoding(d_model, max_seq_len=4096):
    """Return: (L, D)"""
    t = torch.arange(max_seq_len).float()
    freqs = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
    freqs = torch.einsum("i, j -> i j", t, freqs)
    freqs = repeat(freqs, "i j -> i (j r)", r=2)
    return freqs

def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")

@autocast(enabled=False)
def apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2):
    """
    freqs   : [N, D]
    t       : [B, N, D]
    """
    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:].to(t)

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert (
        rot_dim <= t.shape[-1]
    ), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"

    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t_left, t, t_right), dim=-1)


class RoPE(nn.Module):
    def __init__(self, d_model, max_seq_len=4096):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        encoding = get_encoding(d_model, max_seq_len)
        self.register_buffer("encoding", encoding, False)

    def rotate_embedding(self, x):
        """
        x           : [B, N, D]
        rotated_x   : [B, N, D]
        """
        num_views, d_model = x.shape[-2:]

        if num_views > self.max_seq_len :
            encoding = get_encoding(d_model, num_views).to(x)
        else :
            encoding = self.encoding[:num_views]

        rotated_x = apply_rotary_emb(encoding, x, seq_dim=-2)
        return rotated_x
    
if __name__ == "__main__":
    x = torch.rand((1*24, 4, 128))
    pos_encoding = RoPE(d_model=128)

    y = pos_encoding.rotate_embedding(x)
    print(y.shape)