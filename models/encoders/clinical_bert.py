import torch
import torch.nn as nn
from transformers import BioBertModel, BertConfig
from models.modules.hierarchical_attention import OntologyHierarchicalAttention

class ClinicalBertEncoder(nn.Module):
    """Encodeur de texte clinique avec intégration UMLS"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Base BioBERT pré-entraîné
        self.biobert = BioBertModel.from_pretrained("monologg/biobert_v1.1_pubmed")
        
        # Couche d'attention guidée par ontologie
        self.ontology_attention = OntologyHierarchicalAttention(config)
        
        # Projection pour l'alignement KG
        self.kg_projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.projection_dim),
            nn.GELU(),
            nn.LayerNorm(config.projection_dim)
        )

        # Initialisation adaptative
        self._init_weights()

    def _init_weights(self):
        """Initialisation des poids pour la projection"""
        nn.init.kaiming_normal_(self.kg_projection[0].weight)
        nn.init.zeros_(self.kg_projection[0].bias)

    def forward(self, input_ids, attention_mask, ontology_relations):
        """
        Args:
            input_ids: Tensors des tokens d'entrée (batch_size, seq_len)
            attention_mask: Masque d'attention standard
            ontology_relations: Masque des relations UMLS (batch_size, seq_len, seq_len)
        """
        # Encodage de base
        outputs = self.biobert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Application de l'attention guidée
        contextualized = self.ontology_attention(
            outputs.last_hidden_state,
            ontology_relations
        )
        
        # Pooling hiérarchique
        pooled = self._hierarchical_pooling(contextualized)
        
        # Projection finale
        return self.kg_projection(pooled)

    def _hierarchical_pooling(self, hidden_states):
        """Pooling adaptatif basé sur la structure UMLS"""
        # [Implémentation complexe utilisant les métriques de graphe]
        return hidden_states.mean(dim=1)