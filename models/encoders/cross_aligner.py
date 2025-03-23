class MultimodalAligner(nn.Module):
    """Module d'alignement cross-modal avec contraintes UMLS"""
    
    def __init__(self, config):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=config.projection_dim,
            num_heads=config.align_heads
        )
        
        # Projection UMLS-aware
        self.umls_gate = nn.Sequential(
            nn.Linear(2*config.projection_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, text_emb, kg_emb, relation_mask):
        # Attention croisée
        aligned, _ = self.attention(
            text_emb.unsqueeze(1),
            kg_emb.unsqueeze(1),
            kg_emb.unsqueeze(1),
            key_padding_mask=~relation_mask
        )
        
        # Fusion gated
        combined = torch.cat([text_emb, aligned.squeeze(1)], dim=-1)
        gate = self.umls_gate(combined)
        return gate * text_emb + (1 - gate) * aligned.squeeze(1)