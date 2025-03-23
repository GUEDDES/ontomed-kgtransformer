import torch
from sklearn.metrics import f1_score, roc_auc_score

class ClinicalMetrics:
    """Calcule les métriques cliniques avec gestion GPU/CPU"""
    
    def __init__(self):
        self.predictions = []
        self.labels = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        """Ajoute un batch de prédictions"""
        self.predictions.extend(torch.sigmoid(logits).cpu().detach().numpy())
        self.labels.extend(labels.cpu().numpy())

    def compute(self) -> Dict[str, float]:
        """Calcule toutes les métriques"""
        y_pred = np.array(self.predictions) > 0.5
        y_true = np.array(self.labels)
        
        return {
            'accuracy': (y_pred == y_true).mean(),
            'f1_micro': f1_score(y_true, y_pred, average='micro'),
            'auroc': roc_auc_score(y_true, self.predictions)
        }

    def reset(self):
        self.predictions = []
        self.labels = []