import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
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

# Lecture d'un fichier csv
basededonnee = pd.read_csv("arrhythmia.data", names=range(0, 280))
basededonnee = basededonnee.replace(['?'], 'NaN')
print(basededonnee)
# Regarder les données de bd
print(basededonnee.head())
# Résumé statistique description(analyse)
print(basededonnee.describe())

# Taille base de donnée
t = basededonnee.shape

# Les types des colonnes
a = basededonnee.dtypes
# La somme des valeurs manquantes
z = basededonnee.isnull().sum()

# Remplacez les données manquantes par le median de la colonne concernée
num_features = basededonnee.select_dtypes(include=['int64', 'float64']).columns
cat_features = basededonnee.select_dtypes(include=['object']).columns
for i in num_features:
    print(i)
    basededonnee[i].fillna(basededonnee[i].median(), inplace=True)
for i in cat_features:
    print(i)
    basededonnee[i].fillna('None', inplace=True)
# Valeur encoder
scale = LabelEncoder()
for i in cat_features:
    basededonnee[i] = scale.fit_transform(basededonnee[i])
# Analyse des données
# Certaines statistiques descriptives sur notre dataframe:
x = basededonnee.describe()
print(x)
# La moyenne des colonnes de la base de donnée:
n = basededonnee.mean()

A = basededonnee.drop(279, axis=1)
B = basededonnee[279]

# Standardisation
scaler = StandardScaler()
scaler.fit(A)
StandardScaler(copy=True, with_mean=True, with_std=True)
datas = scaler.transform(A)

print(basededonnee.describe())
print(basededonnee.info())

# PCA decomposition
pca = PCA(n_components=16)
pca.fit(datas)
datass = pca.transform(datas)
print(datass.shape)

# K-means (corrigé : boucle pour tester différents k, mais ici fixé à 16 comme dans le code original)
r = range(2, 30)  # Commencer à 2 pour éviter erreurs silhouette
silhouette_coefficients = []
rand = []
for i in r:
    kmeans = KMeans(init="random", n_clusters=16, random_state=42).fit(datas)  # Ajouté random_state pour reproductibilité
    y_kmeans = kmeans.fit_predict(datas)
    centroids = pca.transform(kmeans.cluster_centers_)
    label = kmeans.labels_
    score = silhouette_score(datas, label)
    silhouette_coefficients.append(score)
    ri = rand_score(B, label)
    rand.append(ri)
    plt.scatter(datass[:, 0], datass[:, 1], s=7, c=label)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c="red", s=150)
    plt.title(f'K-Means with 16 clusters (iteration {i})')
    plt.show()

maxs = max(silhouette_coefficients)
mins = min(silhouette_coefficients)
stds = statistics.pstdev(silhouette_coefficients)
moyennes = statistics.mean(silhouette_coefficients)

maxr = max(rand)
minr = min(rand)
stdr = statistics.pstdev(rand)
moyenner = statistics.mean(rand)
print(maxs)
print(mins)
print(maxr)
print(minr)

# Sauvegarde en CSV au lieu d'Excel pour éviter dépendance à xlsxwriter
path = r"C:\Users\Dell\OneDrive\Bureau\MiniProjet\Arrythmie.csv"
dami = {'Silhouette score': [moyennes, stds, mins, maxs], 'RAND index': [moyenner, stdr, minr, maxr]}
out = pd.DataFrame(dami, index=['Moyenne (mean)', 'Ecart type (std)', 'Minimum (worst)', 'Maximum (best)'])
out.to_csv(path)

# Features et target
X = basededonnee.drop(279, axis=1)  # features
Y = basededonnee[279]  # target

# Feature selection
fvalueBest = SelectKBest(score_func=f_classif, k='all')
fvalueBest.fit(X, Y)
names = range(0, 280)
features_score = pd.DataFrame(fvalueBest.scores_)
features = pd.DataFrame(X.columns)
feature_score = pd.concat([features, features_score], axis=1)
feature_score.columns = ["Input_Features", "Score"]
print(feature_score.nlargest(278, columns="Score"))

test = SelectKBest(f_classif, k=20)
fs = test.fit_transform(X, Y)
print("Toute la base ", X.shape)
print("La base réduite ", fs.shape)
scaler = MinMaxScaler()
fs = scaler.fit_transform(fs)

# Listes pour stocker les scores (corrigé : utilisation de fs pour les modèles avec features sélectionnées)
accuracyknn = []
precisionknn = []
recallknn = []
f1_scoreknn = []
accuracybayes = []
precisionbayes = []
recallbayes = []
f1_scorebayes = []
accuracyarbrededecision = []
precisionarbrededecision = []
recallarbrededecision = []
f1_scorearbrededecision = []
accuracysvm = []
precisionsvm = []
recallsvm = []
f1_scoresvm = []

for i in range(30):
    cv = KFold(5, shuffle=True, random_state=42)
    # KNN avec features sélectionnées
    Score1 = cross_val_score(KNeighborsClassifier(), fs, Y, scoring='accuracy', cv=cv)
    Score2 = cross_val_score(KNeighborsClassifier(), fs, Y, scoring=make_scorer(precision_score, average='macro'), cv=cv)
    Score3 = cross_val_score(KNeighborsClassifier(), fs, Y, scoring=make_scorer(recall_score, average='weighted'), cv=cv)
    Score4 = cross_val_score(KNeighborsClassifier(), fs, Y, scoring=make_scorer(f1_score, average='weighted'), cv=cv)

    # Decision Tree avec features sélectionnées
    Score5 = cross_val_score(DecisionTreeClassifier(random_state=42), fs, Y, scoring='accuracy', cv=cv)
    Score6 = cross_val_score(DecisionTreeClassifier(random_state=42), fs, Y, scoring=make_scorer(precision_score, average='macro'), cv=cv)
    Score7 = cross_val_score(DecisionTreeClassifier(random_state=42), fs, Y, scoring=make_scorer(recall_score, average='weighted'), cv=cv)
    Score8 = cross_val_score(DecisionTreeClassifier(random_state=42), fs, Y, scoring=make_scorer(f1_score, average='weighted'), cv=cv)

    # Naive Bayes avec features sélectionnées
    Score9 = cross_val_score(GaussianNB(), fs, Y, scoring='accuracy', cv=cv)
    Score10 = cross_val_score(GaussianNB(), fs, Y, scoring=make_scorer(precision_score, average='macro'), cv=cv)
    Score11 = cross_val_score(GaussianNB(), fs, Y, scoring=make_scorer(recall_score, average='weighted'), cv=cv)
    Score12 = cross_val_score(GaussianNB(), fs, Y, scoring=make_scorer(f1_score, average='weighted'), cv=cv)

    # SVM avec features sélectionnées
    Score13 = cross_val_score(SVC(gamma='auto', random_state=42), fs, Y, scoring='accuracy', cv=cv)
    Score14 = cross_val_score(SVC(gamma='auto', random_state=42), fs, Y, scoring=make_scorer(precision_score, average='macro'), cv=cv)
    Score15 = cross_val_score(SVC(gamma='auto', random_state=42), fs, Y, scoring=make_scorer(recall_score, average='weighted'), cv=cv)
    Score16 = cross_val_score(SVC(gamma='auto', random_state=42), fs, Y, scoring=make_scorer(f1_score, average='weighted'), cv=cv)

    accuracyknn.append(Score1.mean())  # Stocker la moyenne des scores
    precisionknn.append(Score2.mean())
    recallknn.append(Score3.mean())
    f1_scoreknn.append(Score4.mean())

    accuracybayes.append(Score5.mean())
    precisionbayes.append(Score6.mean())
    recallbayes.append(Score7.mean())
    f1_scorebayes.append(Score8.mean())

    accuracyarbrededecision.append(Score9.mean())
    precisionarbrededecision.append(Score10.mean())
    recallarbrededecision.append(Score11.mean())
    f1_scorearbrededecision.append(Score12.mean())

    accuracysvm.append(Score13.mean())
    precisionsvm.append(Score14.mean())
    recallsvm.append(Score15.mean())
    f1_scoresvm.append(Score16.mean())

# Calculs des stats
maxaccknn = np.max(accuracyknn)
minaccknn = np.min(accuracyknn)
moyaccknn = np.mean(accuracyknn)
stdaccknn = np.std(accuracyknn)

maxaccbayes = np.max(accuracybayes)
minaccbayes = np.min(accuracybayes)
moyaccbayes = np.mean(accuracybayes)
stdaccbayes = np.std(accuracybayes)

maxaccarbr = np.max(accuracyarbrededecision)
minaccarbr = np.min(accuracyarbrededecision)
moyaccarbr = np.mean(accuracyarbrededecision)
stdaccarbr = np.std(accuracyarbrededecision)

maxaccsvm = np.max(accuracysvm)
minaccsvm = np.min(accuracysvm)
moyaccsvm = np.mean(accuracysvm)
stdaccsvm = np.std(accuracysvm)

maxpreknn = np.max(precisionknn)
minpreknn = np.min(precisionknn)
moypreknn = np.mean(precisionknn)
stdpreknn = np.std(precisionknn)

maxprebayes = np.max(precisionbayes)
minprebayes = np.min(precisionbayes)
moyprebayes = np.mean(precisionbayes)
stdprebayes = np.std(precisionbayes)

maxprearbr = np.max(precisionarbrededecision)
minprearbr = np.min(precisionarbrededecision)
moyprearbr = np.mean(precisionarbrededecision)
stdprearbr = np.std(precisionarbrededecision)

maxpresvm = np.max(precisionsvm)
minpresvm = np.min(precisionsvm)
moypresvm = np.mean(precisionsvm)
stdpresvm = np.std(precisionsvm)

maxrecknn = np.max(recallknn)
minrecknn = np.min(recallknn)
moyrecknn = np.mean(recallknn)
stdrecknn = np.std(recallknn)

maxrecbayes = np.max(recallbayes)
minrecbayes = np.min(recallbayes)
moyrecbayes = np.mean(recallbayes)
stdrecbayes = np.std(recallbayes)  # Corrigé : était stdprebayes

maxrecarbr = np.max(recallarbrededecision)
minrecarbr = np.min(recallarbrededecision)
moyrecarbr = np.mean(recallarbrededecision)
stdrecarbr = np.std(recallarbrededecision)

maxrecsvm = np.max(recallsvm)
minrecsvm = np.min(recallsvm)
moyrecsvm = np.mean(recallsvm)
stdrecsvm = np.std(recallsvm)

maxf1knn = np.max(f1_scoreknn)
minf1knn = np.min(f1_scoreknn)
moyf1knn = np.mean(f1_scoreknn)
stdf1knn = np.std(f1_scoreknn)

maxf1bayes = np.max(f1_scorebayes)
minf1bayes = np.min(f1_scorebayes)
moyf1bayes = np.mean(f1_scorebayes)
stdf1bayes = np.std(f1_scorebayes)

maxf1arbr = np.max(f1_scorearbrededecision)
minf1arbr = np.min(f1_scorearbrededecision)
moyf1arbr = np.mean(f1_scorearbrededecision)
stdf1arbr = np.std(f1_scorearbrededecision)

maxf1svm = np.max(f1_scoresvm)
minf1svm = np.min(f1_scoresvm)
moyf1svm = np.mean(f1_scoresvm)
stdf1svm = np.std(f1_scoresvm)

# Sauvegarde en CSV
path1 = r"C:\Users\MicroSoft\OneDrive\Bureau\RSD M1\S2\DAMI\Mini Projet\Arrythmieaccuracy.csv"
dami1 = {'KNN': [moyaccknn, stdaccknn, minaccknn, maxaccknn], 'Arbre de décision': [moyaccarbr, stdaccarbr, minaccarbr, maxaccarbr],
         'Bayes': [moyaccbayes, stdaccbayes, minaccbayes, maxaccbayes], 'SVM': [moyaccsvm, stdaccsvm, minaccsvm, maxaccsvm]}
out1 = pd.DataFrame(dami1, index=['Moyenne (mean)', 'Ecart type (std)', 'Minimum (worst)', 'Maximum (best)'])
out1.to_csv(path1)

path2 = r"C:\Users\MicroSoft\OneDrive\Bureau\RSD M1\S2\DAMI\Mini Projet\Arrythmieprecision.csv"
dami2 = {'KNN': [moypreknn, stdpreknn, minpreknn, maxpreknn], 'Arbre de décision': [moyprearbr, stdprearbr, minprearbr, maxprearbr],
         'Bayes': [moyprebayes, stdprebayes, minprebayes, maxprebayes], 'SVM': [moypresvm, stdpresvm, minpresvm, maxpresvm]}
out2 = pd.DataFrame(dami2, index=['Moyenne (mean)', 'Ecart type (std)', 'Minimum (worst)', 'Maximum (best)'])
out2.to_csv(path2)

path3 = r"C:\Users\MicroSoft\OneDrive\Bureau\RSD M1\S2\DAMI\Mini Projet\Arrythmierecall.csv"
dami3 = {'KNN': [moyrecknn, stdrecknn, minrecknn, maxrecknn], 'Arbre de décision': [moyrecarbr, stdrecarbr, minrecarbr, maxrecarbr],
         'Bayes': [moyrecbayes, stdrecbayes, minrecbayes, maxrecbayes], 'SVM': [moyrecsvm, stdrecsvm, minrecsvm, maxrecsvm]}
out3 = pd.DataFrame(dami3, index=['Moyenne (mean)', 'Ecart type (std)', 'Minimum (worst)', 'Maximum (best)'])
out3.to_csv(path3)

path4 = r"C:\Users\MicroSoft\OneDrive\Bureau\RSD M1\S2\DAMI\Mini Projet\Arrythmief1.csv"
dami4 = {'KNN': [moyf1knn, stdf1knn, minf1knn, maxf1knn], 'Arbre de décision': [moyf1arbr, stdf1arbr, minf1arbr, maxf1arbr],
         'Bayes': [moyf1bayes, stdf1bayes, minf1bayes, maxf1bayes], 'SVM': [moyf1svm, stdf1svm, minf1svm, maxf1svm]}
out4 = pd.DataFrame(dami4, index=['Moyenne (mean)', 'Ecart type (std)', 'Minimum (worst)', 'Maximum (best)'])
out4.to_csv(path4)

