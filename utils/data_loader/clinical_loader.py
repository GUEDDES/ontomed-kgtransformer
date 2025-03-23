import pandas as pd
from pathlib import Path
from typing import Tuple

class MIMICLoader:
    """Chargeur de données MIMIC-III avec prétraitement"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.notes_path = self.data_dir / "NOTEEVENTS.csv"
        self.diagnoses_path = self.data_dir / "DIAGNOSES_ICD.csv"

    def load_notes(self) -> pd.DataFrame:
        """Charge et filtre les notes cliniques"""
        notes = pd.read_csv(
            self.notes_path,
            usecols=["ROW_ID", "TEXT", "CATEGORY"],
            dtype={'TEXT': 'string'}
        )
        return notes[notes.CATEGORY.isin(["Discharge summary", "Radiology"])]

    def merge_with_diagnoses(self, notes_df: pd.DataFrame) -> pd.DataFrame:
        """Fusionne avec les codes diagnostiques"""
        diagnoses = pd.read_csv(self.diagnoses_path)
        return pd.merge(
            notes_df,
            diagnoses,
            on="HADM_ID",
            how="inner"
        )