import torch.nn as nn
from .modules import OntologyHierarchicalAttention, DynamicGraphTokenizer
from .encoders import ClinicalBertEncoder, KGTransEEncoder

class OntoMedKGTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Composants principaux
        self.tokenizer = DynamicGraphTokenizer(config)
        self.text_encoder = ClinicalBertEncoder(config)
        self.kg_encoder = KGTransEEncoder(config)
        self.attention = OntologyHierarchicalAttention(config)
        
        # Modules d'alignement
        self.text_proj = nn.Linear(config.text_dim, config.proj_dim)
        self.kg_proj = nn.Linear(config.kg_dim, config.proj_dim)
        self.classifier = nn.Linear(config.proj_dim, config.num_classes)
        
        # Initialisation des poids
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.text_proj.weight)
        nn.init.xavier_uniform_(self.kg_proj.weight)
        
    def forward(self, text_inputs, kg_inputs):
        # Tokenisation dynamique
        tokenized = self.tokenizer(text_inputs)
        
        # Encodage texte avec attention guidée
        text_features = self.text_encoder(
            input_ids=tokenized['input_ids'],
            attention_mask=tokenized['attention_mask']
        )
        text_features = self.attention(text_features, tokenized['relations_mask'])
        
        # Encodage KG
        kg_features = self.kg_encoder(kg_inputs)
        
        # Alignement multimodal
        text_emb = self.text_proj(text_features.mean(dim=1))
        kg_emb = self.kg_proj(kg_features)
        
        # Loss contrastive
        logits = torch.matmul(text_emb, kg_emb.T) / self.config.temperature
        return logits

    def predict(self, text_inputs):
        with torch.no_grad():
            outputs = self.forward(text_inputs)
            return self.classifier(outputs)