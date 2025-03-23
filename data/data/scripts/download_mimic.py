"""
Script pour télécharger et préparer MIMIC-III
Nécessite des accès PhysioNet valides
"""
from pathlib import Path
import requests
import zipfile

MIMIC_URL = "https://physionet.org/files/mimiciii/1.4/"

def download_mimic(output_dir: Path):
    files = [
        "PATIENTS.csv", "ADMISSIONS.csv",
        "NOTEEVENTS.csv", "ICD_DIAGNOSES.csv"
    ]
    
    for file in files:
        print(f"Downloading {file}...")
        r = requests.get(f"{MIMIC_URL}/{file}")
        with open(output_dir/file, 'wb') as f:
            f.write(r.content)
    
    print("Extraction complete.")