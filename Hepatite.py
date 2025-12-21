import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import rand_score
import statistics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import numpy as np
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CHARGEMENT ET PRÉTRAITEMENT DES DONNÉES
# ============================================================================

print("=" * 60)
print("CHARGEMENT ET PRÉTRAITEMENT DES DONNÉES")
print("=" * 60)

basededonnee = pd.read_csv("hepatitis.data", sep='\t', names=range(0, 20))
basededonnee = basededonnee.replace(['?'], np.nan)

print(f"Taille du dataset : {basededonnee.shape}")
print(f"\nAperçu des données :")
print(basededonnee.head())
print(f"\nStatistiques descriptives :")
print(basededonnee.describe())

# Taille base de donnée
t = basededonnee.shape
print(f"\nNombre de patients : {t[0]}")
print(f"Nombre de variables : {t[1]}")

# Les types des colonnes
a = basededonnee.dtypes
print(f"\nTypes des variables :")
print(a)

# La somme des valeurs manquantes
z = basededonnee.isnull().sum()
print(f"\nValeurs manquantes par colonne :")
print(z)

# Gestion des valeurs manquantes
num_features = basededonnee.select_dtypes(include=['int64', 'float64']).columns
cat_features = basededonnee.select_dtypes(include=['object']).columns

print(f"\nImputation des valeurs manquantes...")
for i in num_features:
    basededonnee[i] = basededonnee[i].fillna(basededonnee[i].median())
for i in cat_features:
    basededonnee[i] = basededonnee[i].fillna('None')

# Encodage des variables catégorielles
scale = LabelEncoder()
for i in cat_features:
    basededonnee[i] = scale.fit_transform(basededonnee[i])

# Séparation features ettarget
A = basededonnee.drop(0, axis=1)
B = basededonnee[0]

# Standardisation
scaler = StandardScaler()
datas1 = scaler.fit_transform(A)
print(f"\nDonnées standardisées shape : {datas1.shape}")

# PCA decomposition
n_components_pca = min(16, A.shape[1])
pca = PCA(n_components=n_components_pca)
pca.fit(datas1)
datass = pca.transform(datas1)
print(f"Données après PCA shape : {datass.shape}")


#CLUSTERING 


print("\n" + "=" * 60)
print("ANALYSE DE CLUSTERING")
print("=" * 60)

# Fonction pour déterminer eps optimal pour DBSCAN
def find_optimal_eps(data, min_samples=5):
    """Trouve la valeur optimale de eps pour DBSCAN"""
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(data)
    distances, indices = neighbors_fit.kneighbors(data)
    distances = np.sort(distances[:, min_samples-1], axis=0)
    
    # Trouver le point de coude
    from kneed import KneeLocator
    kneedle = KneeLocator(range(len(distances)), distances, S=1.0, curve='convex', direction='increasing')
    optimal_eps = distances[kneedle.elbow] if kneedle.elbow else distances[-1] * 0.5
    
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.xlabel('Points triés par distance')
    plt.ylabel(f'Distance au {min_samples}ème plus proche voisin')
    plt.title('Méthode du coude pour déterminer eps')
    if kneedle.elbow:
        plt.axvline(x=kneedle.elbow, color='r', linestyle='--', label=f'Coude (eps={optimal_eps:.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return optimal_eps

clustering_results = {
    'KMeans': {'silhouette': [], 'rand': [], 'n_clusters': [], 'labels': None},
    'DBSCAN': {'silhouette': None, 'rand': None, 'n_clusters': None, 'noise': None, 'labels': None},
    'KMedoids': {'silhouette': [], 'rand': [], 'n_clusters': [], 'labels': None}
}


# K-MEANS 


print("\n--- K-Means Clustering (Corrigé) ---")

# Tester différents nombres de clusters
r = range(2, min(11, len(A)//2))  
silhouette_coefficients = []
rand_scores = []
best_kmeans_k = 2
best_kmeans_silhouette = -1
best_kmeans_labels = None
best_kmeans_centroids = None

for n_clusters in r:  
    kmeans = KMeans(init="random", n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(datas1)
    
    if len(set(labels)) > 1:
        score = silhouette_score(datas1, labels)
        ri = rand_score(B, labels)
        
        silhouette_coefficients.append(score)
        rand_scores.append(ri)
        
        clustering_results['KMeans']['silhouette'].append(score)
        clustering_results['KMeans']['rand'].append(ri)
        clustering_results['KMeans']['n_clusters'].append(n_clusters)
        
        if score > best_kmeans_silhouette:
            best_kmeans_silhouette = score
            best_kmeans_k = n_clusters
            best_kmeans_labels = labels
            best_kmeans_centroids = kmeans.cluster_centers_
        
        # Visualisation 
        if n_clusters == best_kmeans_k:
            centroids_pca = pca.transform(kmeans.cluster_centers_)
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(datass[:, 0], datass[:, 1], s=7, c=labels, cmap='tab20')
            plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='*', c="red", s=200, edgecolors='black', linewidths=1.5)
            plt.title(f'K-Means - Meilleur résultat: {n_clusters} clusters (Silhouette: {score:.3f}, RAND: {ri:.3f})')
            plt.xlabel('Première composante principale')
            plt.ylabel('Deuxième composante principale')
            plt.colorbar(scatter, label='Cluster')
            plt.grid(True, alpha=0.3)
            plt.show()

clustering_results['KMeans']['labels'] = best_kmeans_labels

# Statistiques K-Means
if silhouette_coefficients:
    maxs = max(silhouette_coefficients)
    mins = min(silhouette_coefficients)
    stds = statistics.pstdev(silhouette_coefficients)
    moyennes = statistics.mean(silhouette_coefficients)
    
    maxr = max(rand_scores)
    minr = min(rand_scores)
    stdr = statistics.pstdev(rand_scores)
    moyenner = statistics.mean(rand_scores)
    
    print(f"\nMeilleur nombre de clusters K-Means : {best_kmeans_k} (Silhouette: {best_kmeans_silhouette:.3f})")
    print(f"Silhouette scores: Moyenne={moyennes:.3f}, Std={stds:.3f}, Min={mins:.3f}, Max={maxs:.3f}")
    print(f"RAND scores: Moyenne={moyenner:.3f}, Std={stdr:.3f}, Min={minr:.3f}, Max={maxr:.3f}")


# DBSCAN OPTIMISÉ


print("\n--- DBSCAN Clustering (Optimisé) ---")

# Détermination automatique de eps
optimal_eps = find_optimal_eps(datas1, min_samples=5)
print(f"eps optimal déterminé : {optimal_eps:.3f}")

# Test de différents paramètres pour DBSCAN
dbscan_params = [
    {'eps': optimal_eps, 'min_samples': 5},
    {'eps': optimal_eps * 0.8, 'min_samples': 5},
    {'eps': optimal_eps * 1.2, 'min_samples': 5},
    {'eps': optimal_eps, 'min_samples': 3},
    {'eps': optimal_eps, 'min_samples': 7}
]

best_dbscan_score = -1
best_dbscan_labels = None
best_dbscan_params = None

for params in dbscan_params:
    dbscan = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
    labels = dbscan.fit_predict(datas1)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    if n_clusters > 1 and n_clusters < len(datas1)//2:
        score = silhouette_score(datas1[labels != -1], labels[labels != -1]) if np.sum(labels != -1) > 1 else -1
        
        if score > best_dbscan_score:
            best_dbscan_score = score
            best_dbscan_labels = labels
            best_dbscan_params = params
            clustering_results['DBSCAN']['silhouette'] = score
            clustering_results['DBSCAN']['n_clusters'] = n_clusters
            clustering_results['DBSCAN']['noise'] = n_noise
            clustering_results['DBSCAN']['labels'] = labels
            
            if len(set(labels[labels != -1])) > 1:
                ri = rand_score(B[labels != -1], labels[labels != -1])
                clustering_results['DBSCAN']['rand'] = ri

if best_dbscan_labels is not None:
    print(f"\nMeilleurs paramètres DBSCAN : eps={best_dbscan_params['eps']:.3f}, min_samples={best_dbscan_params['min_samples']}")
    print(f"Nombre de clusters : {clustering_results['DBSCAN']['n_clusters']}")
    print(f"Nombre de points bruit : {clustering_results['DBSCAN']['noise']}")
    print(f"Silhouette Score : {clustering_results['DBSCAN']['silhouette']:.3f}" if clustering_results['DBSCAN']['silhouette'] else "Silhouette : N/A")
    print(f"RAND Index : {clustering_results['DBSCAN']['rand']:.3f}" if clustering_results['DBSCAN']['rand'] else "RAND : N/A")
    
    plt.figure(figsize=(10, 6))
    unique_labels = set(best_dbscan_labels)
    colors = [plt.cm.tab20(i) for i in range(len(unique_labels))]
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = 'gray'
            label = 'Bruit'
        else:
            label = f'Cluster {k}'
        
        class_member_mask = (best_dbscan_labels == k)
        xy = datass[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], s=7, c=[col], label=label)
    
    plt.title(f'DBSCAN Clustering (Clusters: {clustering_results["DBSCAN"]["n_clusters"]}, Noise: {clustering_results["DBSCAN"]["noise"]})')
    plt.xlabel('Première composante principale')
    plt.ylabel('Deuxième composante principale')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# K-MEDOIDS


print("\n--- K-Medoids Clustering ---")

r_medoids = range(2, 11)
silhouette_medoids = []
rand_medoids = []
best_k = 2
best_silhouette = -1
best_labels = None
best_medoids_indices = None

for k in r_medoids:
    initial_medoids = list(range(k))
    kmedoids_instance = kmedoids(datas1.tolist(), initial_medoids)
    kmedoids_instance.process()
    clusters = kmedoids_instance.get_clusters()
    medoids_indices = kmedoids_instance.get_medoids()

    labels_medoids = [-1] * len(datas1)
    for cluster_id, cluster in enumerate(clusters):
        for point_idx in cluster:
            labels_medoids[point_idx] = cluster_id

    if len(set(labels_medoids)) > 1:
        score_medoids = silhouette_score(datas1, labels_medoids)
        ri_medoids = rand_score(B, labels_medoids)
        silhouette_medoids.append(score_medoids)
        rand_medoids.append(ri_medoids)
        
        clustering_results['KMedoids']['silhouette'].append(score_medoids)
        clustering_results['KMedoids']['rand'].append(ri_medoids)
        clustering_results['KMedoids']['n_clusters'].append(k)

        if score_medoids > best_silhouette:
            best_silhouette = score_medoids
            best_k = k
            best_labels = labels_medoids
            best_medoids_indices = medoids_indices
    else:
        silhouette_medoids.append(None)
        rand_medoids.append(None)

clustering_results['KMedoids']['labels'] = best_labels

print(f"Meilleur nombre de clusters K-Medoids : {best_k} (Silhouette: {best_silhouette:.3f})")

medoids_pca = pca.transform(datas1[best_medoids_indices])
plt.figure(figsize=(10, 6))
scatter = plt.scatter(datass[:, 0], datass[:, 1], s=7, c=best_labels, cmap='tab20')
plt.scatter(medoids_pca[:, 0], medoids_pca[:, 1], marker='*', c="red", s=200, edgecolors='black', linewidths=1.5)
plt.title(f'K-Medoids avec {best_k} clusters (Silhouette: {best_silhouette:.3f})')
plt.xlabel('Première composante principale')
plt.ylabel('Deuxième composante principale')
plt.colorbar(scatter, label='Cluster')
plt.grid(True, alpha=0.3)
plt.show()

print("\n" + "=" * 60)
print("COMPARAISON DES MÉTHODES DE CLUSTERING")
print("=" * 60)

comparison_data = []

# K-Means 
if clustering_results['KMeans']['labels'] is not None:
    comparison_data.append({
        'Méthode': 'K-Means',
        'N_Clusters': best_kmeans_k,
        'Silhouette': best_kmeans_silhouette,
        'RAND': rand_score(B, clustering_results['KMeans']['labels']),
        'Bruit': 0,
        'Stabilité': np.std(clustering_results['KMeans']['silhouette']) if clustering_results['KMeans']['silhouette'] else 0
    })

# DBSCAN
if clustering_results['DBSCAN']['labels'] is not None:
    comparison_data.append({
        'Méthode': 'DBSCAN',
        'N_Clusters': clustering_results['DBSCAN']['n_clusters'],
        'Silhouette': clustering_results['DBSCAN']['silhouette'] if clustering_results['DBSCAN']['silhouette'] else 'N/A',
        'RAND': clustering_results['DBSCAN']['rand'] if clustering_results['DBSCAN']['rand'] else 'N/A',
        'Bruit': clustering_results['DBSCAN']['noise'],
        'Stabilité': 'N/A'
    })

# K-Medoids
if clustering_results['KMedoids']['labels'] is not None:
    comparison_data.append({
        'Méthode': 'K-Medoids',
        'N_Clusters': best_k,
        'Silhouette': best_silhouette,
        'RAND': rand_score(B, clustering_results['KMedoids']['labels']),
        'Bruit': 0,
        'Stabilité': np.std([s for s in clustering_results['KMedoids']['silhouette'] if s is not None])
    })

comparison_df = pd.DataFrame(comparison_data)
print("\nTableau de comparaison des méthodes de clustering :")
print(comparison_df.to_string(index=False))

# Visualisation comparative
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# K-Means
if clustering_results['KMeans']['labels'] is not None:
    axes[0].scatter(datass[:, 0], datass[:, 1], s=7, c=clustering_results['KMeans']['labels'], cmap='tab20')
    axes[0].set_title(f'K-Means ({best_kmeans_k} clusters)\nSilhouette: {best_kmeans_silhouette:.3f}')
    axes[0].grid(True, alpha=0.3)

# DBSCAN
if clustering_results['DBSCAN']['labels'] is not None:
    axes[1].scatter(datass[:, 0], datass[:, 1], s=7, c=clustering_results['DBSCAN']['labels'], cmap='tab20')
    axes[1].set_title(f'DBSCAN ({clustering_results["DBSCAN"]["n_clusters"]} clusters, {clustering_results["DBSCAN"]["noise"]} bruit)')
    axes[1].grid(True, alpha=0.3)

# K-Medoids
if clustering_results['KMedoids']['labels'] is not None:
    axes[2].scatter(datass[:, 0], datass[:, 1], s=7, c=clustering_results['KMedoids']['labels'], cmap='tab20')
    axes[2].set_title(f'K-Medoids ({best_k} clusters)\nSilhouette: {best_silhouette:.3f}')
    axes[2].grid(True, alpha=0.3)

plt.suptitle('Comparaison des méthodes de clustering', fontsize=16)
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("SAUVEGARDE DES RÉSULTATS")
print("=" * 60)

# Sauvegarde des résultats K-Means
path_kmeans = r"Hepatitis_KMeans_1.csv"
dami_kmeans = {
    'Silhouette score': [moyennes, stds, mins, maxs] if silhouette_coefficients else ['N/A', 'N/A', 'N/A', 'N/A'],
    'RAND index': [moyenner, stdr, minr, maxr] if rand_scores else ['N/A', 'N/A', 'N/A', 'N/A']
}
out_kmeans = pd.DataFrame(dami_kmeans, index=['Moyenne (mean)', 'Ecart type (std)', 'Minimum (worst)', 'Maximum (best)'])
out_kmeans.to_csv(path_kmeans)
print(f"Résultats K-Means sauvegardés dans {path_kmeans}")


path_dbscan = r"Hepatitis_DBSCAN_1.csv"
dami_dbscan = {
    'Silhouette score': [clustering_results['DBSCAN']['silhouette'] if clustering_results['DBSCAN']['silhouette'] else 'N/A'],
    'RAND index': [clustering_results['DBSCAN']['rand'] if clustering_results['DBSCAN']['rand'] else 'N/A'],
    'Clusters': [clustering_results['DBSCAN']['n_clusters'] if clustering_results['DBSCAN']['n_clusters'] else 'N/A'],
    'Noise': [clustering_results['DBSCAN']['noise'] if clustering_results['DBSCAN']['noise'] else 'N/A']
}
out_dbscan = pd.DataFrame(dami_dbscan, index=['Value'])
out_dbscan.to_csv(path_dbscan)
print(f"Résultats DBSCAN sauvegardés dans {path_dbscan}")

path_medoids = r"Hepatitis_KMedoids_1.csv"
valid_silhouettes = [s for s in silhouette_medoids if s is not None]
valid_rands = [r for r in rand_medoids if r is not None]

if valid_silhouettes:
    maxs_medoids = max(valid_silhouettes)
    mins_medoids = min(valid_silhouettes)
    stds_medoids = statistics.pstdev(valid_silhouettes)
    moyennes_medoids = statistics.mean(valid_silhouettes)
else:
    maxs_medoids = mins_medoids = stds_medoids = moyennes_medoids = 'N/A'

if valid_rands:
    maxr_medoids = max(valid_rands)
    minr_medoids = min(valid_rands)
    stdr_medoids = statistics.pstdev(valid_rands)
    moyenner_medoids = statistics.mean(valid_rands)
else:
    maxr_medoids = minr_medoids = stdr_medoids = moyenner_medoids = 'N/A'

dami_medoids = {
    'Silhouette score': [moyennes_medoids, stds_medoids, mins_medoids, maxs_medoids],
    'RAND index': [moyenner_medoids, stdr_medoids, minr_medoids, maxr_medoids]
}
out_medoids = pd.DataFrame(dami_medoids, index=['Moyenne (mean)', 'Ecart type (std)', 'Minimum (worst)', 'Maximum (best)'])
out_medoids.to_csv(path_medoids)
print(f"Résultats K-Medoids sauvegardés dans {path_medoids}")

# Sauvegarde de la comparaison
path_comparison = r"Hepatitis_Clustering_Comparison.csv"
comparison_df.to_csv(path_comparison)
print(f"Comparaison des méthodes sauvegardée dans {path_comparison}")

# CLASSIFICATION 

print("\n" + "=" * 60)
print("CLASSIFICATION ")
print("=" * 60)

X = basededonnee.drop(0, axis=1)
Y = basededonnee[0]


fvalueBest = SelectKBest(score_func=f_classif, k='all')
fvalueBest.fit(X, Y)
features_score = pd.DataFrame(fvalueBest.scores_)
features = pd.DataFrame(X.columns)
feature_score = pd.concat([features, features_score], axis=1)
feature_score.columns = ["Input_Features", "Score"]
print(f"\nTop 10 des features les plus importantes :")
print(feature_score.nlargest(10, columns="Score"))

test = SelectKBest(f_classif, k=10)
fs = test.fit_transform(X, Y)
print(f"\nToute la base : {X.shape}")
print(f"La base réduite : {fs.shape}")

scaler = MinMaxScaler()
fs = scaler.fit_transform(fs)

results = {
    'KNN': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
    'DecisionTree': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
    'NaiveBayes': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []},
    'SVM': {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
}

print("\nÉvaluation des classifieurs (30 répétitions avec random_state différent)...")

for i in range(30):
    cv = KFold(5, shuffle=True, random_state=i)  # CORRECTION : random_state=i au lieu de 42
    
    # KNN
    Score1 = cross_val_score(KNeighborsClassifier(), fs, Y, scoring='accuracy', cv=cv)
    Score2 = cross_val_score(KNeighborsClassifier(), fs, Y, scoring=make_scorer(precision_score, average='macro'), cv=cv)
    Score3 = cross_val_score(KNeighborsClassifier(), fs, Y, scoring=make_scorer(recall_score, average='weighted'), cv=cv)
    Score4 = cross_val_score(KNeighborsClassifier(), fs, Y, scoring=make_scorer(f1_score, average='weighted'), cv=cv)
    
    results['KNN']['accuracy'].append(Score1.mean())
    results['KNN']['precision'].append(Score2.mean())
    results['KNN']['recall'].append(Score3.mean())
    results['KNN']['f1'].append(Score4.mean())
    
    # Decision Tree
    Score5 = cross_val_score(DecisionTreeClassifier(random_state=i), fs, Y, scoring='accuracy', cv=cv)
    Score6 = cross_val_score(DecisionTreeClassifier(random_state=i), fs, Y, scoring=make_scorer(precision_score, average='macro'), cv=cv)
    Score7 = cross_val_score(DecisionTreeClassifier(random_state=i), fs, Y, scoring=make_scorer(recall_score, average='weighted'), cv=cv)
    Score8 = cross_val_score(DecisionTreeClassifier(random_state=i), fs, Y, scoring=make_scorer(f1_score, average='weighted'), cv=cv)
    
    results['DecisionTree']['accuracy'].append(Score5.mean())
    results['DecisionTree']['precision'].append(Score6.mean())
    results['DecisionTree']['recall'].append(Score7.mean())
    results['DecisionTree']['f1'].append(Score8.mean())
    
    # Naive Bayes
    Score9 = cross_val_score(GaussianNB(), fs, Y, scoring='accuracy', cv=cv)
    Score10 = cross_val_score(GaussianNB(), fs, Y, scoring=make_scorer(precision_score, average='macro'), cv=cv)
    Score11 = cross_val_score(GaussianNB(), fs, Y, scoring=make_scorer(recall_score, average='weighted'), cv=cv)
    Score12 = cross_val_score(GaussianNB(), fs, Y, scoring=make_scorer(f1_score, average='weighted'), cv=cv)
    
    results['NaiveBayes']['accuracy'].append(Score9.mean())
    results['NaiveBayes']['precision'].append(Score10.mean())
    results['NaiveBayes']['recall'].append(Score11.mean())
    results['NaiveBayes']['f1'].append(Score12.mean())
    
    # SVM
    Score13 = cross_val_score(SVC(gamma='auto', random_state=i), fs, Y, scoring='accuracy', cv=cv)
    Score14 = cross_val_score(SVC(gamma='auto', random_state=i), fs, Y, scoring=make_scorer(precision_score, average='macro'), cv=cv)
    Score15 = cross_val_score(SVC(gamma='auto', random_state=i), fs, Y, scoring=make_scorer(recall_score, average='weighted'), cv=cv)
    Score16 = cross_val_score(SVC(gamma='auto', random_state=i), fs, Y, scoring=make_scorer(f1_score, average='weighted'), cv=cv)
    
    results['SVM']['accuracy'].append(Score13.mean())
    results['SVM']['precision'].append(Score14.mean())
    results['SVM']['recall'].append(Score15.mean())
    results['SVM']['f1'].append(Score16.mean())

# Calcul des statistiques
classification_stats = []

for model_name, metrics in results.items():
    classification_stats.append({
        'Modèle': model_name,
        'Accuracy (mean)': np.mean(metrics['accuracy']),
        'Accuracy (std)': np.std(metrics['accuracy']),
        'Precision (mean)': np.mean(metrics['precision']),
        'Recall (mean)': np.mean(metrics['recall']),
        'F1-Score (mean)': np.mean(metrics['f1'])
    })

classification_df = pd.DataFrame(classification_stats)
print("\nPerformance des classifieurs:")
print(classification_df.to_string(index=False))

# Visualisation des résultats
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
models = list(results.keys())
colors = ['blue', 'green', 'orange', 'red']

for idx, (model, color) in enumerate(zip(models, colors)):
    row, col = divmod(idx, 2)
    
    axes[row, col].hist(results[model]['accuracy'], bins=15, alpha=0.7, color=color, edgecolor='black')
    axes[row, col].axvline(np.mean(results[model]['accuracy']), color='red', linestyle='--', 
                           label=f'Moyenne: {np.mean(results[model]["accuracy"]):.3f}')
    axes[row, col].set_title(f'{model} - Accuracy Distribution')
    axes[row, col].set_xlabel('Accuracy')
    axes[row, col].set_ylabel('Fréquence')
    axes[row, col].legend()
    axes[row, col].grid(True, alpha=0.3)

plt.suptitle('Distribution des scores de précision (30 répétitions)', fontsize=16)
plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("ANALYSE TERMINÉE AVEC SUCCÈS")
print("=" * 60)
