import pandas as pd

# Chargement des données
df = pd.read_csv("HR_comma_sep.csv")

# Aperçu
df.head()
print(df.head)
import matplotlib.pyplot as plt
import seaborn as sns

# Taille du dataset
print("Taille du dataset :", df.shape)

# Vérifier les types de données et les valeurs manquantes
print(df.info())

# Statistiques descriptives
print(df.describe())

# Répartition des employés qui ont quitté ou non
sns.countplot(x='left', data=df)
plt.title("Répartition employés restés vs partis")
plt.xlabel("left (0 = resté, 1 = parti)")
plt.ylabel("Nombre d'employés")
plt.show()
# Encodage des variables catégorielles
df_encoded = pd.get_dummies(df, columns=['salary', 'Department'], drop_first=True)

# Renommer la colonne cible
df_encoded.rename(columns={'left': 'target'}, inplace=True)

# Vérifier le résultat
df_encoded.head()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Séparer features et target
X = df_encoded.drop('target', axis=1)
y = df_encoded['target']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entraîner le modèle
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Évaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# Importance des variables
importances = model.feature_importances_
features = X.columns

# Visualisation
plt.figure(figsize=(10,6))
sns.barplot(x=importances, y=features)
plt.title("Importance des variables")
plt.show()
