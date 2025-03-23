# OntoMed-KGTransformer

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

A neuro-symbolic framework for clinical knowledge graph fusion.

## Features
- Ontology-guided hierarchical attention
- Dynamic graph-aware tokenization
- UMLS-integrated contrastive learning

## Installation

git clone https://github.com/gueddes/ontomed-kgtransformer.git
cd ontomed-kgtransformer
pip install -r requirements.txt


## Usage

from models import OntoMedKGTransformer

config = load_config("configs/base.yaml")
model = OntoMedKGTransformer(config)

## Structure

ontomed-kgtransformer/
├── data/
│   ├── processors/
│   │   ├── mimic_processor.py      # Prétraitement MIMIC-III
│   │   └── kg_processor.py         # Chargement Hetionet/UMLS
│   └── sample_data/                # Données de test
├── models/
│   ├── modules/
│   │   ├── hierarchical_attention.py # Attention hiérarchique
│   │   ├── graph_tokenizer.py       # Tokenisation dynamique
│   │   └── graph_pe.py             # Encodage positionnel graphe
│   ├── encoders/
│   │   ├── clinical_bert.py        # Encodeur texte amélioré
│   │   └── kg_encoder.py           # Encodeur KG avec TransE
│   ├── losses/
│   │   └── ontology_contrastive.py # Loss contrastive avancée
│   └── model.py                    # Architecture complète
├── utils/
│   ├── umls_tools/
│   │   ├── relation_checker.py     # Vérification relations UMLS
│   │   └── cui_embedding.py        # Gestion embeddings UMLS
│   ├── graph_ops/
│   │   ├── neighbor_sampler.py     # Échantillonnage voisins KG
│   │   └── centrality.py           # Calcul métriques de centralité
│   └── optim/
│       └── lr_scheduler.py         # Planificateur taux d'apprentissage
├── configs/
│   └── train_config.yaml           # Configuration hyperparamètres
├── scripts/
│   ├── preprocess_data.py          # Script prétraitement
│   └── run_training.py             # Pipeline complet d'entraînement
├── requirements.txt                # Dépendances détaillées
└── README.md                       # Documentation technique
