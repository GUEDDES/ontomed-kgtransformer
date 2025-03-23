import yaml
from pathlib import Path
from typing import Any, Dict

class ConfigManager:
    """Gestionnaire centralisé de configuration"""
    
    def __init__(self, config_path: str = "configs/base.yaml"):
        self.config = self._load_config(config_path)
        self._validate()

    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(Path(path), 'r') as f:
            return yaml.safe_load(f)

    def _validate(self):
        """Valide la structure de la configuration"""
        required_keys = {
            'data', 'model', 'training', 
            'paths', 'umls_relations'
        }
        missing = required_keys - set(self.config.keys())
        if missing:
            raise KeyError(f"Configuration incomplète. Manque: {missing}")

    def get(self, key: str, default: Any = None) -> Any:
        """Accès sécurisé aux paramètres"""
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except KeyError:
            return default