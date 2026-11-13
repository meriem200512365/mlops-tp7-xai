
import mlflow

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)

from xai_commun import VERSION, m, Xte, yte, CAS

# Recuperation du run d'origine
run = mlflow.get_run(VERSION.run_id)

# Predictions
pred = m.predict(Xte)
proba = m.predict_proba(Xte)[:, 1]

print("Modele : fraude, version", VERSION.version)
print("Run :", VERSION.run_id[:8])

print("Classe :", type(m).__name__)

print(
    "Parametres :",
    {
        k: run.data.params[k]
        for k in ["n_estimators", "max_depth"]
    }
)

print("F1 :", round(f1_score(yte, pred), 4))
print("F1 enregistre :", round(run.data.metrics["f1"], 4))

print("Precision :", round(precision_score(yte, pred), 4))
print("Rappel :", round(recall_score(yte, pred), 4))

print("AUC :", round(roc_auc_score(yte, proba), 4))

print("Matrice de confusion :")
print(confusion_matrix(yte, pred))

print(
    "Mme Benali : P(fraude) =",
    round(m.predict_proba(CAS)[0, 1], 4)
)
