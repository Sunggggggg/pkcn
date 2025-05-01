import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output

class EntropyLoss(nn.Module):
    def __init__(self, eps = 1e-12):
        super(EntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, x):
        b = x * torch.log(x + self.eps)
        b = -1.0 * b.sum(dim=1)
        b = b.mean()
        return b

class MemoryUnit(nn.Module):
    def __init__(self, mem_dim, fea_dim, shrink_thres=0.0025):
        super(MemoryUnit, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres
        self.weight = nn.Parameter(torch.Tensor(self.mem_dim, self.fea_dim))  # M x C
        self.reset_parameters()

    def reset_parameters(self):
        self.bias = None
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """
        input : [B, dim]
        """
        att_weight = F.linear(input, self.weight)  
        
        if(self.shrink_thres>0):
            att_weight = F.softmax(att_weight, dim=1) 
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)

        mem_trans = self.weight.permute(1, 0)
        output = F.linear(att_weight, mem_trans)
        return output, att_weight

class MemoryModule(nn.Module):
    def __init__(self, cat_type, mem_dim, fea_dim, shrink_thres=0.0025, device='cuda'):
        super(MemoryModule, self).__init__()
        self.mem_dim = mem_dim
        self.fea_dim = fea_dim
        self.shrink_thres = shrink_thres

        sub_dim = fea_dim // cat_type
        self.appear_category_layers = nn.ModuleList([
            nn.Linear(fea_dim, sub_dim) for _ in range(cat_type)
        ])

        self.memory = MemoryUnit(self.mem_dim, sub_dim, self.shrink_thres)

    def forward(self, x):
        """
        x : [B, 400]
        return
        appear_feat_list: [B, 400]
        att_weight_list : [B, N, 256]
        """
        appear_feat_list, att_weight_list = [], []
        for layers in self.appear_category_layers:
            appear_feat = layers(x)
            appear_feat, att_weight = self.memory(appear_feat)
            appear_feat_list.append(appear_feat)
            att_weight_list.append(att_weight)

        appear_feat_list = torch.cat(appear_feat_list, dim=-1)  
        att_weight_list = torch.stack(att_weight_list, dim=-1)  
        
        return appear_feat_list, att_weight_list

if __name__ == "__main__":
    x = torch.rand((1, 400))
    model = MemoryModule(4, 256, 400)
    result_feat = model(x)
