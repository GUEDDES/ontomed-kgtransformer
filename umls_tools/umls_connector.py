import requests
from typing import Dict, List

class UMLSConnector:
    """Client pour l'API UMLS avec gestion du cache"""
    
    def __init__(self, api_key: str, version: str = "2023AA"):
        self.base_url = "https://uts-ws.nlm.nih.gov/rest"
        self.api_key = api_key
        self.version = version
        self.cache = {}

    def get_cui_relations(self, cui: str, relation_types: List[str]) -> Dict:
        """Récupère les relations UMLS pour un CUI donné"""
        cache_key = f"{cui}_{'_'.join(relation_types)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        url = f"{self.base_url}/content/{self.version}/CUI/{cui}/relations"
        params = {
            'apiKey': self.api_key,
            'relType': ','.join(relation_types)
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        relations = response.json()["result"]
        self.cache[cache_key] = relations
        return relations

    def cui_to_label(self, cui: str) -> str:
        """Convertit un CUI en libellé textuel"""
        url = f"{self.base_url}/content/{self.version}/CUI/{cui}"
        params = {'apiKey': self.api_key}
        
        response = requests.get(url, params=params)
        return response.json()["name"]