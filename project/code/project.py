import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.dummy as skd
import sklearn.model_selection as skm

features=pd.read_csv("../DRIAMS-EC/driams_Escherichia_coli_Ceftriaxone_features.csv")
labels=pd.read_csv("../DRIAMS-EC/driams_Escherichia_coli_Ceftriaxone_labels.csv")
data=features.merge(labels)

def endless_plotter(data):
    fig, axes=plt.subplots(nrows=600,ncols=10,figsize=(35,400))
    for i in range(0,data.shape[1]-2):
        sns.histplot(data=data,x=data.iloc[:,i],y=data["label"],ax=axes[int(i/10)][int(i%10)])
        print(i)
    print("saving figure, this might take a while...")
    plt.savefig("../output/plot1")
    print("figure saved!")

def checkna(data):
    datana=data.isna()
    print(f"dataset has {np.sum(np.sum(datana,axis=1))} NA")

## I DONT LIKE NAMES 0798f178-95d5-4e92-ad63-831cf50605b5_MALDI2, I WANT MALDI2, THAT'S IT, THEIR NAME WILL BE THEIR INDEX. ACTUALLY, M1=0, M2=1
def maldiRename(data):
    data = data.rename(columns={"Unnamed: 0" : "type"})
    M1 = np.sum(data.loc[:,"type"].str.contains("MALDI1"))
    M2 = np.sum(data.loc[:,"type"].str.contains("MALDI2"))
    data.loc[data.loc[:,"type"].str.contains("MALDI1"), "type"] = "M1"
    data.loc[data.loc[:,"type"].str.contains("MALDI2"), "type"] = "M2"
    print(f"There are: M1 {M1}, M2 {M2}")
    return data

def createDummies(data):
    dummies=pd.get_dummies(data["type"],drop_first=True)
    dummies.reset_index(inplace=True)
    data.reset_index(inplace=True)
    data=data.merge(dummies,on="index",how="left")
    data=data.drop(["type","index"],axis=1)
    data["M2"]=data["M2"].astype(float)
    return data

##  MAYBE TYPE EXPLAINS THE LABEL SOME WAY, actually no... maybe M2 has slightly more probability of label 1, but not much is seen.
##  PLOT M1 - M2 vs label

def m1_m2_plotter(data):
    fig = plt.figure(figsize=(6,6)) 
    sns.countplot(data=data,x="label",hue="type")
    plt.savefig("../output/M1-M2 vs label")

###CHECK FOR DOUBLES
def drop_duplicates(data):
    if data.equals(data.drop_duplicates()):
        print("dataset has no duplicates")
    else:
        data=data.drop_duplicates()
        print("dataset had duplicates and they have been dropped")
    return data

###HEATMAP PLOTTER
def heatmapPlotter(data):
    corr=data.corr()
    plt.figure(figsize=(300,300))
    sns.heatmap(corr)
    plt.savefig("../output/heatmap")
    print("plotted")

def splitData(data):
    tune, test = skm.train_test_split(data, test_size=0.15)
    tune.reset_index(inplace=True)
    tune.drop(axis=1, inplace=True,columns="index")
    test.reset_index(inplace=True)
    test.drop(axis=1, inplace=True,columns="index")
    print("data splitted")
    print(f"tune: {tune.shape}\n test: {test.shape}")
    return tune, test

### K - FOLD:    SPLITTING INTO 5, ACCESSIBLE THROUGH LIST, SHUFFLE TRUE
### K - FOLD:    SPLITTING INTO 5, ACCESSIBLE THROUGH LIST, SHUFFLE TRUE
def kfolderino(data):
    traininglist = []
    validatelist = []
    kfolder = skm.KFold(n_splits = 5,shuffle=True,random_state=2025)
    for train, validate in kfolder.split(tune):
        traininglist.append(tune.iloc[train,:])
        validatelist.append(tune.iloc[validate,:])
    print("K-fold success")
    return validatelist, traininglist

### kNN
def knn_with_corr_filter(data, thresholds=None, k_list=None, cv=5):
    """
    Führt kNN-Klassifikation für verschiedene Korrelationsschwellen und k-Werten durch.
    Filtert vorab nicht-numerische Features heraus.
    Gibt DataFrame mit allen Ergebnissen und die beste Parameter-Kombination zurück.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)
    if k_list is None:
        k_list = list(range(1, 21, 2))

    # Merkmale und Label trennen
    X = data.drop("label", axis=1)
    y = data["label"]

    # Nur numerische Features für Korrelationsanalyse
    X_numeric = X.select_dtypes(include=[np.number])
    dropped = X.shape[1] - X_numeric.shape[1]
    if dropped > 0:
        print(f"Hinweis: {dropped} nicht-numerische Features wurden vor der Korrelation entfernt.")

    # Korrelation der numerischen Features mit dem Label berechnen
    correlations = X_numeric.apply(lambda col: col.corr(y))
    results = []

    for thresh in thresholds:
        # Features mit absoluter Korrelation >= Schwelle auswählen
        selected = correlations[correlations.abs() >= thresh].index.tolist()
        if not selected:
            continue
        X_sel = X_numeric[selected]
        for k in k_list:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = skm.cross_val_score(knn, X_sel, y, cv=cv)
            results.append({"threshold": thresh, "k": k, "mean_score": scores.mean()})

    # Ergebnisse zusammenfassen
    results_df = pd.DataFrame(results)
    if results_df.empty:
        raise ValueError("Keine Kombination von Features und Parametern gefunden. Schwellenwerte oder Daten prüfen.")

    # Beste Parameter-Kombination ermitteln
    best = results_df.loc[results_df["mean_score"].idxmax()]
    return results_df, best

# Beispielaufruf der neuen Funktion
results_df, best_params = knn_with_corr_filter(data)
print("Beste Parameter:", best_params)
print(results_df.sort_values("mean_score", ascending=False).head(10))



###     USING FUNCTIONS
checkna(data)
data=drop_duplicates(data)
data=maldiRename(data)
data=createDummies(data)
tune, test = splitData(data)
validatelist, traininglist = kfolderino(data)


