import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.dinov2.models.vision_transformer import vit_base
from options import opts

def freeze_model(m):
    m.requires_grad_(False)

def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)

class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.opts = opts

        self.dino = vit_base(patch_size=14, block_chunks=0, init_values=1.0) 
        self.dino.load_state_dict(torch.load('./dinov2_vitb14_pretrain.pth', map_location=self.device))
        self.dino.apply(freeze_all_but_bn)

        # Prompt Learning
        self.sk_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))
        self.img_prompt = nn.Parameter(torch.randn(self.opts.n_prompts, self.opts.prompt_dim))

        self.distance_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y)
        self.loss_fn_triplet = nn.TripletMarginWithDistanceLoss(
             distance_function=self.distance_fn, margin=0.2)

        self.best_metric = -1e3

    def configure_optimizers(self):
        model_params = list(self.dino.parameters())

        optimizer = torch.optim.Adam([
            {'params': model_params, 'lr': self.opts.clip_LN_lr},
            {'params': [self.sk_prompt] + [self.img_prompt], 'lr': self.opts.prompt_lr}])
        return optimizer
    

    def forward(self, data, dtype='image'):
        if dtype == 'image':
            feat = self.dino(data, prompt=self.img_prompt.expand(data.shape[0], -1, -1))
        else:
            # dtype == 'sketch'
            feat = self.dino(data, prompt=self.sk_prompt.expand(data.shape[0], -1, -1))
        return feat