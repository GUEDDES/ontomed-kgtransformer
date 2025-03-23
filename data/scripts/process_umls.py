"""
Pipeline de traitement UMLS avec filtres médicaux
"""
import pandas as pd
from typing import Dict

def filter_umls(conso_path: str, rel_path: str, config: Dict) -> pd.DataFrame:
    # Chargement des données brutes
    columns = ["CUI", "LAT", "TS", "STT",...] # 18 colonnes RRF
    df_conso = pd.read_csv(conso_path, sep='|', names=columns, dtype='str')
    
    # Filtrage selon la configuration
    medical_concepts = df_conso[
        (df_conso.SAB.isin(config["sources"])) &
        (df_conso.LAT == "ENG")
    ]
    
    # Traitement des relations
    rel_columns = ["CUI1", "REL", "CUI2", ...]
    df_rel = pd.read_csv(rel_path, sep='|', names=rel_columns)
    
    return medical_concepts, df_rel