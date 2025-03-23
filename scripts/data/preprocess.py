# scripts/data/preprocess.py
import pandas as pd
from utils.data_processor import MIMICProcessor, KGProcessor

def main():
    # Traitement des données cliniques
    mimic_proc = MIMICProcessor(config.mimic_path)
    clinical_data = mimic_proc.load_notes().clean().tokenize()
    
    # Traitement du Knowledge Graph
    kg_proc = KGProcessor(config.kg_path)
    kg = kg_proc.load_graph().prune().add_umls_relations()
    
    # Sauvegarde
    clinical_data.save("data/processed/clinical.feather")
    kg.save("data/processed/kg_graph.bin")