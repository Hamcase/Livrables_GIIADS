# 📊 M5 Forecasting Data

## ⚠️ Fichiers de données non inclus

Les fichiers de données du projet M5 ne sont **pas inclus dans ce repository** car ils dépassent la limite de taille de GitHub (100 MB).

### Fichiers requis :
- `sales_train_validation.csv` (114 MB)
- `sell_prices.csv` (194 MB)
- `calendar.csv` (inclus - 60 KB)
- `sample_submission.csv` (inclus - 44 MB)

## 📥 Comment obtenir les données

### Option 1 : Kaggle (Recommandé)
1. Créez un compte sur [Kaggle](https://www.kaggle.com/)
2. Téléchargez le dataset : [M5 Forecasting - Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy/data)
3. Placez les fichiers dans ce dossier `m5-forecasting-data/`

### Option 2 : Kaggle API
```bash
# Installez l'API Kaggle
pip install kaggle

# Configurez vos credentials (fichier kaggle.json)
# Téléchargez le dataset
kaggle competitions download -c m5-forecasting-accuracy

# Extrayez les fichiers
unzip m5-forecasting-accuracy.zip -d m5-forecasting-data/
```

### Option 3 : Utiliser les données réduites
Le dossier `../reduced_data/` contient des versions réduites des datasets pour le prototypage :
- `sales_df_reduced.csv`
- `prices_df_reduced.csv`
- `calendar_df_reduced.csv`

Ces fichiers sont inclus dans le repository et permettent de tester le code sans télécharger les données complètes.

## 📝 Structure attendue

Après téléchargement, la structure doit être :
```
m5-forecasting-data/
├── calendar.csv                     ✅ Inclus
├── sales_train_validation.csv       ❌ À télécharger (114 MB)
├── sell_prices.csv                  ❌ À télécharger (194 MB)
└── sample_submission.csv            ✅ Inclus
```

## 🔧 Modification du notebook

Si vous utilisez les données réduites, modifiez les chemins dans le notebook :
```python
# Au lieu de :
sales_df = pd.read_csv('m5-forecasting-data/sales_train_validation.csv')
prices_df = pd.read_csv('m5-forecasting-data/sell_prices.csv')

# Utilisez :
sales_df = pd.read_csv('reduced_data/sales_df_reduced.csv')
prices_df = pd.read_csv('reduced_data/prices_df_reduced.csv')
calendar_df = pd.read_csv('reduced_data/calendar_df_reduced.csv')
```

## ℹ️ Informations sur le dataset

**M5 Forecasting Competition** - Walmart Sales Forecasting
- **30,490 produits** dans 10 magasins
- **1,913 jours** d'historique de ventes
- **3 états** : CA, TX, WI
- **Hiérarchie** : État → Magasin → Catégorie → Département → Produit
