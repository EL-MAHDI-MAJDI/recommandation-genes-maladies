# 🧬 Recommandation des Maladies-Gènes par Prédiction de Liens

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NetworkX](https://img.shields.io/badge/NetworkX-Graph%20Mining-FF6B35?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Node2Vec](https://img.shields.io/badge/Node2Vec-Embeddings-9B59B6?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-27AE60?style=for-the-badge)

**Système de recommandation gène-maladie basé sur la prédiction de liens dans un graphe biparti**

*Module : Graph Mining & Intelligence Artificielle*

</div>

---

## 📋 Table des Matières

- [Vue d'ensemble](#-vue-densemble)
- [Résultats Clés](#-résultats-clés)
- [Architecture du Projet](#-architecture-du-projet)
- [Installation](#-installation)
- [Pipeline en 7 Phases](#-pipeline-en-7-phases)
- [Données](#-données)
- [Méthodes Implémentées](#-méthodes-implémentées)
- [Résultats Détaillés](#-résultats-détaillés)
- [Utilisation](#-utilisation)
- [Structure des Fichiers](#-structure-des-fichiers)
- [Limitations et Perspectives](#-limitations-et-perspectives)

---

## 🎯 Vue d'ensemble

Ce projet applique les techniques de **Graph Mining** et d'**Intelligence Artificielle** pour prédire automatiquement des associations entre gènes et maladies encore inconnues de la science.

### Le problème

Identifier quel gène cause quelle maladie prend en moyenne **10 à 20 ans** de recherche expérimentale. Les bases de données biologiques (comme HPO) contiennent des associations déjà validées, mais restent **incomplètes** : de nombreux liens biologiques réels n'ont pas encore été découverts.

### Notre approche

On modélise les associations gène-maladie comme un **graphe biparti** :

```
G = (U ∪ V, E)

U = ensemble des Gènes   🧬
V = ensemble des Maladies 🦠
E = associations validées scientifiquement
```

La tâche : pour chaque paire **(gène, maladie)** non encore reliée, prédire un **score de probabilité** entre 0 et 1 indiquant si une association biologique existe probablement.

---

## 🏆 Résultats Clés

| Métrique | Valeur |
|---|---|
| **Meilleur modèle** | GAE + Random Forest |
| **AUC-ROC (IA)** | **0.6956** |
| **AUC-ROC (Baseline)** | 0.5751 |
| **Gain IA vs Baseline** | **+12.05 points** |
| **Associations dans le graphe** | 15 940 |
| **Gènes uniques** | ~4 500 |
| **Maladies uniques** | ~8 000 |

### Exemple de recommandations produites

**Maladie D_ORPHA:528084 → Top 5 gènes candidats :**

| Rang | Gène | Score IA | Confiance |
|---|---|---|---|
| 1 | WASHC5 | 0.9867 | 🔴 TRÈS HAUTE |
| 2 | CCDC22 | 0.9867 | 🔴 TRÈS HAUTE |
| 3 | VPS35L | 0.9867 | 🔴 TRÈS HAUTE |
| 4 | BRPF1  | 0.9867 | 🔴 TRÈS HAUTE |
| 5 | SLC2A3 | 0.9733 | 🔴 TRÈS HAUTE |

**Gène HBB → Top 5 maladies candidates :**

| Rang | Maladie | Score IA | Confiance |
|---|---|---|---|
| 1 | ORPHA:330032 | 0.9000 | 🔴 TRÈS HAUTE |
| 2 | ORPHA:619233 | 0.7733 | 🔴 TRÈS HAUTE |
| 3 | OMIM:617101  | 0.7733 | 🔴 TRÈS HAUTE |
| 4 | ORPHA:280615 | 0.7600 | 🔴 TRÈS HAUTE |
| 5 | OMIM:613977  | 0.7600 | 🔴 TRÈS HAUTE |

---

## 🏗️ Architecture du Projet

```
recommandation-genes-maladies/
│
├── 📁 data/
│   ├── raw/                          # Données brutes HPO téléchargées
│   │   └── genes_to_disease.txt      # Fichier source HPO
│   ├── processed/                    # Données nettoyées
│   │   ├── gene_disease_edges.csv    # Arêtes du graphe (gene_id, disease_id)
│   │   ├── gene_disease_full.csv     # Données complètes avec symboles
│   │   ├── genes_list.csv            # Liste des gènes uniques
│   │   └── diseases_list.csv         # Liste des maladies uniques
│   ├── graph/                        # Graphe NetworkX sérialisé
│   │   ├── gene_disease_graph.pkl    # Graphe complet
│   │   ├── gene_disease_graph.graphml# Format Gephi/Cytoscape
│   │   └── node_types.csv            # Types des nœuds
│   ├── splits/                       # Données train/test
│   │   ├── train_data.csv            # Exemples d'entraînement (80%)
│   │   ├── test_data.csv             # Exemples de test (20%)
│   │   ├── G_train.pkl               # Graphe d'entraînement
│   │   └── pos/neg_train/test_edges  # Arêtes positives/négatives
│   └── embeddings/
│       └── node2vec_embeddings.pkl   # Vecteurs Node2Vec calculés
│
├── 📁 notebooks/                     # Jupyter Notebooks par phase
│   ├── Phase2_Collecte_Donnees.ipynb
│   ├── Phase3_Construction_Graphe.ipynb
│   ├── Phase4_Preparation_TrainTest.ipynb
│   ├── Phase5_Methodes_Baseline.ipynb
│   ├── Phase6_Methodes_IA.ipynb
│   └── Phase7_Systeme_Recommandation.ipynb
│
├── 📁 results/
│   ├── baseline_scores.csv           # Scores des méthodes baseline
│   ├── baseline_metrics.csv          # AUC, AP, P@K des baselines
│   ├── final_comparison.csv          # Comparaison toutes méthodes
│   ├── best_model.pkl                # Meilleur modèle sauvegardé
│   ├── final_model_deployed.pkl      # Modèle final déployé
│   ├── figures/                      # Tous les graphiques générés
│   │   ├── degree_distributions.png
│   │   ├── graphe_biparti_viz.png
│   │   ├── baseline_roc_curves.png
│   │   ├── ai_vs_baseline_roc.png
│   │   └── ...
│   └── recommendations/              # Recommandations produites
│       ├── rec_genes_example.csv
│       ├── rec_diseases_example.csv
│       └── top5_diseases_recommendations.csv
│
└── 📄 README.md
```

---

## ⚙️ Installation

### Prérequis

- Python 3.10+
- pip

### Étape 1 — Cloner le dépôt

```bash
git clone https://github.com/votre-username/recommandation-genes-maladies.git
cd recommandation-genes-maladies
```

### Étape 2 — Installer les dépendances

```bash
pip install pandas numpy networkx matplotlib seaborn requests
pip install scikit-learn scipy node2vec gensim
pip install jupyter
```

### Étape 3 — Lancer Jupyter

```bash
jupyter notebook
```

Ouvrir les notebooks dans le dossier `notebooks/` dans l'ordre des phases (Phase2 → Phase7).

---

## 🔄 Pipeline en 7 Phases

```
Phase 1          Phase 2          Phase 3          Phase 4
Cadrage    ──►   Données    ──►   Graphe     ──►   Train/Test
Théorique        HPO              NetworkX          Split 80/20

Phase 5          Phase 6          Phase 7
Baseline   ──►   IA (GAE)   ──►   Système de
Heuristiques     Node2Vec         Recommandation
```

### Phase 1 — Compréhension et Cadrage
Définition de la problématique comme tâche de **prédiction de liens** dans un graphe biparti gène-maladie. Revue de la littérature, choix des méthodes, périmètre du projet.

### Phase 2 — Collecte et Préparation des Données
- Téléchargement automatique de `genes_to_disease.txt` depuis HPO
- Correction du format d'identifiants (`NCBIGene:64170` → `64170`)
- Nettoyage pandas : suppression doublons, valeurs manquantes, IDs invalides
- Production de 4 fichiers CSV propres

### Phase 3 — Construction et Analyse du Graphe
- Construction du graphe biparti avec NetworkX
- Vérification de la propriété bipartite
- Analyse structurelle : degrés, hubs, composantes connexes, densité
- Sauvegarde en formats `.pkl` et `.graphml`

### Phase 4 — Préparation Train/Test
- Division 80/20 des arêtes existantes (exemples positifs)
- Génération d'exemples négatifs (paires sans lien connu, ratio 1:1)
- Validation avec **5 tests qualité** anti data-leakage
- Résultat : 25 504 exemples train, 6 376 exemples test

### Phase 5 — Méthodes Baseline (Heuristiques)

> ⚠️ **Découverte importante** : les formules classiques (CN, Jaccard...) donnent score=0 sur un graphe biparti car N(gène) ∩ N(maladie) = ∅. On a adapté les formules aux voisins du 2ème degré N²(G).

6 méthodes implémentées et évaluées :

| Méthode | AUC-ROC | Note |
|---|---|---|
| Preferential Attachment | 0.3704 | Pire que le hasard ! |
| Score Combiné | 0.3784 | - |
| Resource Allocation Biparti | 0.5750 | - |
| Common Neighbors Biparti | 0.5751 | - |
| Adamic-Adar Biparti | 0.5751 | - |
| **Jaccard Biparti** | **0.5751** | **Meilleure baseline** |

### Phase 6 — Méthodes d'Intelligence Artificielle

**Node2Vec** (marches aléatoires → embeddings 64D → classifieur) :
- AUC = 0.34–0.40 ❌ (échec dû à la fragmentation du graphe)
- Variance PCA des embeddings = 6.9% seulement

**GAE — Graph Autoencoder** (factorisation SVD → embeddings 64D → classifieur) :
- Logistic Regression : AUC = 0.5608
- **Random Forest : AUC = 0.6956** ✅ **(meilleur modèle)**

### Phase 7 — Système de Recommandation Final
- Réentraînement du modèle GAE+RF sur 100% des données
- Classe `GeneDiseasRecommender` avec deux fonctions :
  - `recommend_genes_for_disease(disease_id, top_k)` → top-K gènes
  - `recommend_diseases_for_gene(gene_node, top_k)` → top-K maladies
- Score de confiance en 4 niveaux (Très Haute / Haute / Moyenne / Faible)

---

## 📦 Données

### Source

**Human Phenotype Ontology (HPO)**
- URL : [hpo.jax.org](https://hpo.jax.org)
- Fichier : `genes_to_disease.txt`
- Accès : gratuit, sans inscription, mis à jour régulièrement
- Types d'associations : MENDELIAN, POLYGENIC

### Statistiques

```
Lignes brutes téléchargées  : 15 941
Après nettoyage             : 15 940
Gènes uniques               : ~4 500
Maladies uniques            : ~8 000
Densité du graphe           : 0.000440
Sources des maladies        : OMIM (~60%), ORPHA (~40%)
```

### Correction appliquée

```python
# Problème : format "NCBIGene:64170"
# Solution :
df['gene_id'] = df['ncbi_gene_id'].str.replace('NCBIGene:', '', regex=False)
```

---

## 🔬 Méthodes Implémentées

### Méthodes Baseline (Phase 5)

Les formules classiques de prédiction de liens ont été **adaptées au graphe biparti** en utilisant les voisins du 2ème degré :

```python
# N²(gène G) = gènes qui partagent une maladie avec G
def get_gene_neighbors_L2(G, gene_node):
    diseases = set(G.neighbors(gene_node))
    genes_L2 = set()
    for d in diseases:
        genes_L2.update(G.neighbors(d))
    genes_L2.discard(gene_node)
    return genes_L2

# Common Neighbors Biparti
# CN(G, D) = |N²(G) ∩ N(D)|
def common_neighbors_bipartite(G, gene_node, disease_node):
    genes_L2 = get_gene_neighbors_L2(G, gene_node)
    genes_of_disease = set(G.neighbors(disease_node))
    return float(len(genes_L2 & genes_of_disease))

# Jaccard Biparti
# J(G, D) = |N²(G) ∩ N(D)| / |N²(G) ∪ N(D)|
def jaccard_bipartite(G, gene_node, disease_node):
    genes_L2 = get_gene_neighbors_L2(G, gene_node)
    genes_of_disease = set(G.neighbors(disease_node))
    union = genes_L2 | genes_of_disease
    if len(union) == 0:
        return 0.0
    return len(genes_L2 & genes_of_disease) / len(union)
```

### GAE — Graph Autoencoder (Phase 6)

```python
from scipy.sparse import lil_matrix
from sklearn.decomposition import TruncatedSVD

# Construction matrice d'adjacence
A = lil_matrix((N, N), dtype=np.float32)
for u, v in G_train.edges():
    i, j = node_to_idx[u], node_to_idx[v]
    A[i, j] = A[j, i] = 1.0

# Décomposition SVD : A ≈ U × Σ × Vᵀ
svd = TruncatedSVD(n_components=64, random_state=42)
embeddings_matrix = svd.fit_transform(A.tocsr())

# Prédiction via produit Hadamard + Random Forest
def get_edge_features(u, v, embeddings):
    return embeddings[u] * embeddings[v]   # Hadamard
```

---

## 📊 Résultats Détaillés

### Comparaison toutes méthodes

| Rang | Méthode | Catégorie | AUC-ROC |
|---|---|---|---|
| 🥇 1 | GAE + Random Forest | IA | **0.6956** |
| 2 | Jaccard Biparti | Baseline | 0.5751 |
| 3 | CN Biparti | Baseline | 0.5751 |
| 4 | Adamic-Adar Biparti | Baseline | 0.5751 |
| 5 | GAE + Logistic Regression | IA | 0.5608 |
| 6 | Node2Vec + Random Forest | IA | 0.4025 |
| 7 | Node2Vec + Log. Reg. | IA | 0.3436 |

### Observations scientifiques importantes

**1. Pourquoi Node2Vec a échoué**

Le graphe HPO est très fragmenté : ~92% des nœuds ont un degré ≤ 1 dans le graphe d'entraînement. Les marches aléatoires de Node2Vec ne peuvent pas explorer efficacement ce type de graphe (elles restent bloquées sur les nœuds isolés). La variance expliquée par PCA des embeddings n'est que de 6.9%, confirmant leur faible qualité informationnelle.

**2. Pourquoi GAE a réussi là où Node2Vec a échoué**

La factorisation SVD opère **globalement** sur toute la matrice d'adjacence en une seule opération mathématique. Même pour un nœud peu connecté, le vecteur résultant capture sa position relative dans la structure globale du graphe, sans avoir besoin de chemins existants pour l'atteindre.

**3. Biais vers les nœuds de faible degré**

Le modèle favorise statistiquement les paires impliquant des nœuds rares (degré=1). Ceci est cohérent avec la biologie des maladies mendéliennes (causées par des gènes très spécifiques) et explique le score AUC=0.37 du Preferential Attachment.

---

## 🚀 Utilisation

### Charger le système et faire des recommandations

```python
import pickle
import numpy as np
import pandas as pd

# Charger le modèle déployé
with open('results/final_model_deployed.pkl', 'rb') as f:
    artifact = pickle.load(f)

model      = artifact['model']
embeddings = artifact['embeddings']

# Fonction de prédiction
def get_edge_features(u, v, emb):
    if u not in emb or v not in emb:
        return np.zeros(64)
    return emb[u] * emb[v]   # Hadamard

# Recommander des gènes pour une maladie
def recommend_genes(disease_id, G_full, top_k=10):
    gene_nodes = [n for n in G_full.nodes()
                  if G_full.nodes[n].get('node_type') == 'gene']
    known = set(G_full.neighbors(disease_id))
    candidates = [g for g in gene_nodes
                  if g not in known and g in embeddings]
    pairs  = [(disease_id, g) for g in candidates]
    X      = np.array([get_edge_features(u, v, embeddings) for u, v in pairs])
    scores = model.predict_proba(X)[:, 1]
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(candidates[i], round(float(scores[i]), 4)) for i in top_idx]

# Exemple d'utilisation
resultats = recommend_genes('D_ORPHA:528084', G_full, top_k=5)
for gene, score in resultats:
    print(f"  {gene}  →  score = {score}")
```

### Lancer les notebooks dans l'ordre

```bash
# Phase 2 — Collecte des données
jupyter notebook notebooks/Phase2_Collecte_Donnees.ipynb

# Phase 3 — Construction du graphe
jupyter notebook notebooks/Phase3_Construction_Graphe.ipynb

# Phase 4 — Préparation Train/Test
jupyter notebook notebooks/Phase4_Preparation_TrainTest.ipynb

# Phase 5 — Méthodes Baseline
jupyter notebook notebooks/Phase5_Methodes_Baseline.ipynb

# Phase 6 — IA (Node2Vec + GAE)
# ⚠️ Node2Vec peut prendre 20-30 min selon votre machine
jupyter notebook notebooks/Phase6_Methodes_IA.ipynb

# Phase 7 — Système de Recommandation
jupyter notebook notebooks/Phase7_Systeme_Recommandation.ipynb
```

---

## 📁 Structure des Fichiers Produits

Après exécution complète du pipeline, les fichiers suivants sont disponibles :

```
results/
├── baseline_metrics.csv          # Tableau AUC de toutes les baselines
├── baseline_scores.csv           # Scores individuels par paire de test
├── final_comparison.csv          # Comparaison Baseline vs IA
├── best_model.pkl                # Meilleur modèle Phase 6
├── final_model_deployed.pkl      # Modèle final Phase 7
└── recommendations/
    ├── rec_genes_example.csv           # Top-10 gènes pour la maladie exemple
    ├── rec_diseases_example.csv        # Top-10 maladies pour le gène exemple
    └── top5_diseases_recommendations.csv  # Recommandations pour 5 maladies
```

---

## ⚠️ Limitations et Perspectives

### Limitations actuelles

| Limitation | Explication |
|---|---|
| **Échec Node2Vec** | Graphe trop fragmenté pour les marches aléatoires |
| **Biais degré faible** | Modèle favorise les nœuds peu connectés |
| **Négatifs bruités** | Certains "non-liens" pourraient être des associations réelles non découvertes |
| **Évaluation locale** | AUC mesuré sur des paires, pas sur la cohérence biologique des recommandations |

### Pistes d'amélioration

- **Enrichir le graphe** avec les interactions protéine-protéine (base STRING) pour densifier les connexions et permettre à Node2Vec de fonctionner
- **Correction du biais de degré** en ajoutant `deg(u)` et `deg(v)` comme features explicites dans le classifieur
- **GNN (Graph Neural Network)** avec PyTorch Geometric (GraphSAGE, GCN) pour une représentation encore plus riche
- **Validation biologique** des recommandations en comparant avec des publications récentes
- **Interface utilisateur** pour interroger le système sans code

---

## 📚 Références

- Human Phenotype Ontology (HPO) : [hpo.jax.org](https://hpo.jax.org)
- Node2Vec : Grover & Leskovec, *node2vec: Scalable Feature Learning for Networks*, KDD 2016
- Graph Autoencoder : Kipf & Welling, *Variational Graph Auto-Encoders*, 2016
- Prédiction de liens dans les réseaux biologiques : Hristova et al., *Gene-disease association prediction*, Bioinformatics 2021
- NetworkX : [networkx.org](https://networkx.org)
- Scikit-learn : [scikit-learn.org](https://scikit-learn.org)

---

## 📄 Licence

Ce projet est réalisé dans le cadre d'un module universitaire de Graph Mining & Intelligence Artificielle.

---

<div align="center">

**🧬 Projet Graph Mining — Recommandation Gènes-Maladies**

*Pipeline complet : HPO → Graphe Biparti → GAE → Système de Recommandation*

**AUC Baseline : 0.5751 → AUC IA : 0.6956 → Gain : +12.05 points**

</div>