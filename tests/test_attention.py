import torch
from models.modules.hierarchical_attention import OntologyHierarchicalAttention

def test_attention_mechanism():
    config = MockConfig()
    attention = OntologyHierarchicalAttention(config)
    
    x = torch.randn(2, 128, 768)  # Batch x SeqLen x HiddenDim
    mask = torch.randint(0, 2, (2, 128, 128))  # Random relations mask
    
    output = attention(x, mask)
    assert output.shape == x.shape, "Attention output shape mismatch"