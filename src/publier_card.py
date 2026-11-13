from pathlib import Path
import mlflow
from mlflow import MlflowClient

# Configuration du serveur de suivi MLflow
mlflow.set_tracking_uri('http://127.0.0.1:5000')
client = MlflowClient()

# Récupération de la version du modèle en production et de l'ID du run XAI
version_obj = client.get_model_version_by_alias('fraude', 'production')
run_id = Path('reports/run_xai.txt').read_text().strip()  # Run de l'activité 6

# Réouverture et enrichissement du run existant avec la Model Card
with mlflow.start_run(run_id=run_id):
    mlflow.log_artifact('MODEL_CARD.md')

# Lecture de la Model Card pour mise à jour dans le registre
card_content = Path('MODEL_CARD.md').read_text(encoding='utf-8')
client.update_model_version('fraude', version_obj.version, description=card_content)

# Ajout d'un tag de traçabilité sur la version du modèle
client.set_model_version_tag(
    'fraude',
    version_obj.version,
    'model_card',
    f'MODEL_CARD.md, run {run_id[:8]}',
)

print(
    f'Model Card attachee a fraude v{version_obj.version} '
    f'et au run {run_id[:8]}'
)