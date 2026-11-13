
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow import MlflowClient
from sklearn.model_selection import train_test_split

# 1. Connexion au serveur MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# 2. Nom du modele et alias
NOM = "fraude"
ALIAS = "production"

# 3. Recuperation de la version en production
VERSION = MlflowClient().get_model_version_by_alias(
    NOM, ALIAS
)

# 4. Chargement du modele
m = mlflow.sklearn.load_model(
    f"models:/{NOM}@{ALIAS}"
)

# 5. Chargement des donnees
df = pd.read_csv("data/transactions.csv")

X = df.drop(columns="fraude")
y = df["fraude"]

# 6. Meme decoupage que TP02
Xtr, Xte, ytr, yte = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# 7. Transaction de Mme Benali
CAS = pd.DataFrame([
    {
        "montant": 1800.0,
        "anciennete": 30,
        "freq_24h": 4,
        "nuit": 1
    }
])

print("Modele charge :", NOM)
print("Version :", VERSION.version)
print("Run d'origine :", VERSION.run_id)
print("Dimensions :", X.shape)
print("Transaction de Benali :")
print(CAS)
