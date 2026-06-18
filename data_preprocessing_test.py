"""
Phase 2 — Nettoyage et préparation des données
Auteur : Projet Graph Mining - Recommandation Gènes-Maladies
"""

import pandas as pd
import json
import os

# ============================================================
# CHEMINS
# ============================================================
RAW_PATH      = "/home/claude/recommandation-genes-maladies/data/raw/genes_to_disease.txt"
PROCESSED_DIR = "/home/claude/recommandation-genes-maladies/data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_data(path):
    """Charge le fichier brut TSV gène-maladie"""
    df = pd.read_csv(path, sep="\t")
    print(f"✅ Données chargées : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    print(f"   Colonnes : {list(df.columns)}\n")
    return df

def explore_data(df):
    """Analyse exploratoire rapide des données brutes"""
    print("=" * 55)
    print("  EXPLORATION DES DONNÉES BRUTES")
    print("=" * 55)

    print(f"\n📌 Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"\n📌 Types de colonnes :")
    print(df.dtypes.to_string())

    print(f"\n📌 Valeurs manquantes par colonne :")
    print(df.isnull().sum().to_string())

    print(f"\n📌 Statistiques clés :")
    print(f"   Gènes uniques    : {df['gene_id'].nunique()}")
    print(f"   Maladies uniques : {df['disease_id'].nunique()}")
    print(f"   Associations     : {len(df)}")
    print(f"   Doublons         : {df.duplicated().sum()}")

    print(f"\n📌 Top 5 gènes les plus associés à des maladies :")
    top_genes = df['gene_id'].value_counts().head(5)
    for gene, count in top_genes.items():
        print(f"   {gene:10s} → {count} maladies")

    print(f"\n📌 Top 5 maladies avec le plus de gènes associés :")
    top_diseases = df.groupby(['disease_id','disease_name']).size().sort_values(ascending=False).head(5)
    for (did, dname), count in top_diseases.items():
        print(f"   {dname:45s} → {count} gènes")

    return df

def clean_data(df):
    """Nettoyage : suppression des doublons, valeurs manquantes, colonnes inutiles"""
    print("\n" + "=" * 55)
    print("  NETTOYAGE DES DONNÉES")
    print("=" * 55)

    initial_size = len(df)

    # 1. Suppression des doublons
    df = df.drop_duplicates(subset=["gene_id", "disease_id"])
    print(f"\n✅ Doublons supprimés : {initial_size - len(df)} lignes")

    # 2. Suppression des valeurs manquantes
    df = df.dropna(subset=["gene_id", "disease_id"])
    print(f"✅ Valeurs manquantes supprimées")

    # 3. Nettoyage des espaces dans les noms
    df["gene_id"]      = df["gene_id"].str.strip().str.upper()
    df["disease_id"]   = df["disease_id"].str.strip()
    df["gene_name"]    = df["gene_name"].str.strip()
    df["disease_name"] = df["disease_name"].str.strip()
    print(f"✅ Nettoyage des espaces effectué")

    # 4. Réinitialiser l'index
    df = df.reset_index(drop=True)

    print(f"\n📌 Données après nettoyage : {len(df)} associations")
    return df

def create_edge_list(df):
    """
    Crée la liste d'arêtes finale pour construire le graphe.
    Chaque ligne = une arête (gene_id, disease_id)
    """
    # Préfixer les IDs pour distinguer gènes et maladies dans le graphe
    edge_list = df[["gene_id", "disease_id", "gene_name", "disease_name"]].copy()
    edge_list["node_gene"]    = "G_" + edge_list["gene_id"]
    edge_list["node_disease"] = "D_" + edge_list["disease_id"]

    # Sauvegarder la liste d'arêtes
    edge_list_path = os.path.join(PROCESSED_DIR, "edge_list.csv")
    edge_list[["node_gene", "node_disease", "gene_name", "disease_name"]].to_csv(
        edge_list_path, index=False
    )

    # Sauvegarder aussi les nœuds avec leurs labels
    genes_nodes = edge_list[["node_gene","gene_name"]].drop_duplicates()
    genes_nodes.columns = ["node_id","label"]
    genes_nodes["type"] = "gene"

    disease_nodes = edge_list[["node_disease","disease_name"]].drop_duplicates()
    disease_nodes.columns = ["node_id","label"]
    disease_nodes["type"] = "disease"

    all_nodes = pd.concat([genes_nodes, disease_nodes], ignore_index=True)
    all_nodes.to_csv(os.path.join(PROCESSED_DIR, "nodes.csv"), index=False)

    print(f"\n✅ Liste d'arêtes sauvegardée : {edge_list_path}")
    print(f"   {len(edge_list)} arêtes au total")
    print(f"\nAperçu de la liste d'arêtes :")
    print(edge_list[["node_gene","node_disease","gene_name","disease_name"]].head(8).to_string())

    return edge_list

def generate_summary(df):
    """Génère un résumé JSON du dataset"""
    summary = {
        "nombre_genes": int(df["gene_id"].nunique()),
        "nombre_maladies": int(df["disease_id"].nunique()),
        "nombre_associations": int(len(df)),
        "densite_graphe": round(len(df) / (df["gene_id"].nunique() * df["disease_id"].nunique()), 4),
        "top_genes": df["gene_id"].value_counts().head(5).to_dict(),
        "top_maladies": df.groupby("disease_name").size().sort_values(ascending=False).head(5).to_dict()
    }
    summary_path = os.path.join(PROCESSED_DIR, "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Résumé sauvegardé : {summary_path}")
    print(f"\n📊 Densité du graphe : {summary['densite_graphe']}")
    print(f"   (proportion des liens existants sur tous les liens possibles)")
    return summary

# ============================================================
# PIPELINE PRINCIPAL
# ============================================================
if __name__ == "__main__":
    print("\n🚀 PHASE 2 — Préparation des données\n")

    df = load_data(RAW_PATH)
    df = explore_data(df)
    df = clean_data(df)
    edge_list = create_edge_list(df)
    summary = generate_summary(df)

    # Sauvegarder le fichier nettoyé
    clean_path = os.path.join(PROCESSED_DIR, "genes_diseases_clean.csv")
    df.to_csv(clean_path, index=False)

    print(f"\n✅ Données nettoyées sauvegardées : {clean_path}")
    print("\n🎉 Phase 2 terminée avec succès !\n")