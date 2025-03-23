import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from utils import OntologyAwareSampler
from models import OntoMedKGTransformer
from losses import OntologyContrastiveLoss

def train(config):
    # Initialisation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OntoMedKGTransformer(config).to(device)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    # Chargement des données
    train_dataset = ClinicalDataset(config.data_path, split='train')
    train_sampler = OntologyAwareSampler(train_dataset, config.relations_map)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn
    )
    
    # Loss et métriques
    criterion = OntologyContrastiveLoss(config.temperature, config.margin)
    
    # Boucle d'entraînement
    for epoch in range(config.epochs):
        model.train()
        for batch in train_loader:
            texts, kg_data, labels = batch
            texts = texts.to(device)
            kg_data = kg_data.to(device)
            
            # Forward pass
            text_emb, kg_emb = model(texts, kg_data)
            
            # Calcul de la loss
            loss = criterion(text_emb, kg_emb, labels)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            
        # Validation
        model.eval()
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        
    # Sauvegarde finale
    torch.save(model.state_dict(), f'{config.save_dir}/final_model.pth')