import torch
import torch.nn as nn
import torch.nn.functional as F

class OntologyContrastiveLoss(nn.Module):
    """Loss contrastive guidée par ontologie avec hard negative mining"""
    
    def __init__(self, config):
        super().__init__()
        self.temperature = config.temperature
        self.margin = config.margin
        self.negatives_per_positive = config.negatives_per_positive

    def forward(self, text_emb, kg_emb, ontology_matrix):
        """
        Args:
            text_emb: Embeddings texte (batch_size, dim)
            kg_emb: Embeddings KG (batch_size, dim)
            ontology_matrix: Matrice des relations UMLS (batch_size, batch_size)
        """
        # Similarités calculées
        sim_matrix = torch.matmul(text_emb, kg_emb.T) / self.temperature
        
        # Génération des étiquettes positives
        positive_mask = ontology_matrix > 0.7  # Seuil de similarité sémantique
        negatives_mask = ontology_matrix < 0.3
        
        # Extraction des paires positives
        pos_sim = sim_matrix[positive_mask]
        
        # Hard negative mining
        neg_sim = sim_matrix[negatives_mask]
        topk_neg = torch.topk(neg_sim, k=self.negatives_per_positive, largest=True).values
        
        # Calcul de la loss
        pos_loss = -torch.log(torch.exp(pos_sim)).mean()
        neg_loss = -torch.log(1 - torch.exp(topk_neg)).mean()
        
        # Combinaison avec marge
        return torch.clamp(pos_loss + neg_loss + self.margin, min=0.0)

    def _generate_hard_negatives(self, embeddings):
        """Génération d'échantillons difficiles via mixup"""
        # [Implémentation avancée utilisant MixUp et les relations UMLS]
        return augmented_negatives