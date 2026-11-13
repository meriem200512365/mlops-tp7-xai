import hashlib
from pathlib import Path
import mlflow
from mlflow import MlflowClient
from sklearn.metrics import f1_score, precision_score, recall_score
from xai_commun import ALIAS, NOM, VERSION, Xte, m, yte


def empreinte(chemin):
    """Calcule l'empreinte MD5 d'un fichier (même principe que DVC)."""
    return hashlib.md5(Path(chemin).read_bytes()).hexdigest()


# 1. Génération des prédictions
pred = m.predict(Xte)

# 2. Configuration de l'expérience MLflow
mlflow.set_experiment('tp7-explicabilite')

with mlflow.start_run(run_name=f'xai-{NOM}-v{VERSION.version}') as run:
    # Enregistrement des tags descriptifs
    mlflow.set_tags(
        {
            'modele': NOM,
            'alias': ALIAS,
            'modele_version': VERSION.version,
            'modele_run_id': VERSION.run_id,
            'donnees': 'data/transactions.csv',
            'donnees_md5': empreinte('data/transactions.csv'),
            'methodes': 'permutation ; SHAP TreeExplainer ; LIME',
        }
    )

    # Enregistrement des paramètres
    mlflow.log_params(
        {
            'shap_echantillon': 400,
            'lime_num_samples': 2000,
            'lime_random_state': 0,
            'permutation_n_repeats': 10,
        }
    )

    # Enregistrement des métriques de performance
    mlflow.log_metrics(
        {
            'f1': f1_score(yte, pred),
            'precision': precision_score(yte, pred),
            'recall': recall_score(yte, pred),
        }
    )

    # Archivage des rapports générés (graphiques, LIME, JSON)
    for f in sorted(Path('reports').glob('*.*')):
        if f.suffix not in ('.png', '.html', '.json'):
            continue
        mlflow.log_artifact(str(f), artifact_path='explications')

    # Sauvegarde locale de l'ID du run
    Path('reports/run_xai.txt').write_text(run.info.run_id)

# 3. Association du run d'explication au versioning du modèle
MlflowClient().set_model_version_tag(
    NOM, VERSION.version, 'explications_run', run.info.run_id
)

print(
    f'run {run.info.run_id[:8]} : explications du modele {NOM} '
    f'v{VERSION.version} archivees'
)