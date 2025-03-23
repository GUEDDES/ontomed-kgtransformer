import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_

class TransEKGEncoder(nn.Module):
    """Encodeur de graphe basé sur TransE avec métriques de graphe"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings des entités et relations
        self.entity_emb = nn.Embedding(config.num_entities, config.kg_embed_dim)
        self.relation_emb = nn.Embedding(config.num_relations, config.kg_embed_dim)
        
        # Module de centralité
        self.centrality_encoder = nn.Embedding(config.centrality_bins, config.kg_embed_dim)
        
        # Initialisation
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialisation des embeddings selon le protocole TransE"""
        xavier_normal_(self.entity_emb.weight)
        xavier_normal_(self.relation_emb.weight)
        nn.init.uniform_(self.centrality_encoder.weight, -0.1, 0.1)

    def forward(self, triplets, centrality_indices):
        """
        Args:
            triplets: Tensors (h, r, t) shape (batch_size, 3)
            centrality_indices: Indices de centralité discrétisés (batch_size,)
        """
        h = self.entity_emb(triplets[:,0])
        r = self.relation_emb(triplets[:,1])
        t = self.entity_emb(triplets[:,2])
        
        # Encodage TransE standard
        transE_scores = torch.norm(h + r - t, p=1, dim=1)
        
        # Incorporation de la centralité
        centrality = self.centrality_encoder(centrality_indices)
        
        # Combinaison non linéaire
        combined = torch.sigmoid(transE_scores.unsqueeze(1) + centrality)
        
        return combined

    def get_entity_embeddings(self):
        """Récupère les embeddings pour l'alignement contrastif"""
        return self.entity_emb.weight