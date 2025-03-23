# scripts/evaluate.py
from models import OntoMedKGTransformer
from utils.metrics import ClinicalMetrics

def evaluate(model, test_loader):
    model.eval()
    metrics = ClinicalMetrics()
    
    for batch in test_loader:
        texts, kg_data, labels = batch
        outputs = model.predict(texts)
        metrics.update(outputs, labels)
        
    return {
        'accuracy': metrics.accuracy(),
        'f1_micro': metrics.f1_score(micro=True),
        'auroc': metrics.auroc()
    }