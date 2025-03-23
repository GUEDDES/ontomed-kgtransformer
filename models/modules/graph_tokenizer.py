import networkx as nx
from quickumls import QuickUMLS
from transformers import BertTokenizer

class DynamicGraphTokenizer:
    def __init__(self, config):
        self.umls_matcher = QuickUMLS(
            config.umls_path,
            overlapping_criteria='length',
            threshold=0.7
        )
        self.bert_tokenizer = BertTokenizer.from_pretrained(config.bert_model)
        self.kg_graph = nx.read_gpickle(config.kg_path)
        self.max_neighbors = config.max_neighbors
        
        # Initialiser les métriques de graphe
        self.node_centrality = nx.betweenness_centrality(self.kg_graph)

    def _get_graph_positional_encoding(self, cui):
        centrality = self.node_centrality.get(cui, 0.0)
        degree = self.kg_graph.degree(cui)
        return [centrality, degree]

    def tokenize(self, text):
        # Étape 1: Extraction des entités UMLS
        matches = self.umls_matcher.match(text)
        cuis = list({match['cui'] for match in matches})
        
        # Étape 2: Tokenisation BERT standard
        tokens = self.bert_tokenizer(
            text,
            add_special_tokens=True,
            max_length=config.max_seq_len,
            truncation=True
        )
        
        # Étape 3: Incorporation des entités KG
        kg_tokens = []
        for cui in cuis:
            neighbors = list(nx.neighbors(self.kg_graph, cui))[:self.max_neighbors]
            for neighbor in neighbors:
                kg_tokens.append(f"[KG_{neighbor}]")
                # Ajout de l'encodage positionnel
                pe = self._get_graph_positional_encoding(neighbor)
                kg_tokens.append(f"[PE_{'_'.join(map(str, pe))}]")
        
        # Combinaison des tokens
        full_tokens = tokens + kg_tokens
        return self.bert_tokenizer.encode(
            full_tokens,
            padding='max_length',
            max_length=config.max_seq_len,
            return_tensors='pt'
        )