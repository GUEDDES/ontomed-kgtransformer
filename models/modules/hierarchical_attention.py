import torch
from torch import nn, einsum
from einops import rearrange

class OntologyHierarchicalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(config.hidden_size, 3*config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.relation_emb = nn.Embedding(100, self.head_dim)  # 100 relations UMLS types
        
    def apply_ontology_constraints(self, attn, relations_mask):
        # Applique le masque hiérarchique
        hierarchy_levels = relations_mask.float().unsqueeze(1)
        attn = attn * hierarchy_levels + (1 - hierarchy_levels) * -1e9
        return attn

    def forward(self, x, relations_mask):
        B, N, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        
        # Calcul de l'attention avec relations UMLS
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        relation_bias = self.relation_emb(relations_mask)  # [B, N, N, d]
        relation_bias = rearrange(relation_bias, 'b i j d -> b () i j d')
        dots = dots + einsum('b h i j, b i j d -> b h i j d', dots, relation_bias).mean(-1)
        
        # Application des contraintes hiérarchiques
        attn = self.apply_ontology_constraints(dots, relations_mask)
        attn = attn.softmax(dim=-1)
        
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)