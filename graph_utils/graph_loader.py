import networkx as nx
import pickle
from typing import Union

def load_knowledge_graph(file_path: str, fmt: str = "gpickle") -> nx.MultiDiGraph:
    """Charge un graphe depuis différents formats"""
    loaders = {
        "gpickle": nx.read_gpickle,
        "graphml": nx.read_graphml,
        "edgelist": lambda p: nx.read_edgelist(p, create_using=nx.MultiDiGraph)
    }
    
    if fmt not in loaders:
        raise ValueError(f"Format {fmt} non supporté. Options: {list(loaders.keys())}")
    
    try:
        return loaders[fmt](file_path)
    except Exception as e:
        raise RuntimeError(f"Échec du chargement du graphe: {str(e)}")

def save_kg_embeddings(embeddings: dict, path: str):
    """Sauvegarde les embeddings de KG avec format optimisé"""
    with open(path, 'wb') as f:
        pickle.dump({
            'metadata': {
                'num_nodes': len(embeddings),
                'embed_dim': len(next(iter(embeddings.values())))
            },
            'embeddings': embeddings
        }, f, protocol=4)