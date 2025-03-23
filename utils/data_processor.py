# utils/data_processor.py
import pyarrow.feather as feather
import networkx as nx

class MIMICProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        
    def load_notes(self):
        self.df = pd.read_csv(f"{self.data_path}/NOTES.csv")
        return self
    
    def clean(self):
        self.df = self.df.dropna(subset=['TEXT'])
        return self

class KGProcessor:
    def __init__(self, kg_path):
        self.kg_path = kg_path
        
    def load_graph(self):
        self.graph = nx.read_gpickle(f"{self.kg_path}/graph.pkl")
        return self
    
    def prune(self, min_degree=2):
        nodes_to_keep = [n for n,d in self.graph.degree() if d >= min_degree]
        self.graph = self.graph.subgraph(nodes_to_keep)
        return self