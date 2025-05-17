import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.dummy as skd
import sklearn.model_selection as skm
import sklearn.feature_selection as fs
import sklearn.linear_model as sklm
import time
import sklearn.ensemble as ske
import sklearn.metrics as skmtr
import sklearn.svm as svm
import sklearn.neighbors as skn

start_time = time.time()



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

def checkImbalance(data):
    print(f"Data is unbalanced: {data["label"].sum(axis=0)/data.shape[0]} resistant bacteria sample and {1-data["label"].sum(axis=0)/data.shape[0]} non resistant bacteria samples")

def train_random_forest(tune_df, cv_splits=5, random_state=2025):
    """
    Tune a RandomForestClassifier on ⁠ tune_df ⁠.
    Assumes ⁠ tune_df ⁠ has a 'label' column and all other columns are numeric features.
    Returns the best‐found RF model.
    """
    # Split X / y
    X_tune = tune_df.drop(columns=["label"])
    y_tune = tune_df["label"].astype(int)
    
    # Parameter grid
    param_grid = {
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
    }
    
    # RandomForest with balanced class weights
    rf = ske.RandomForestClassifier(
        random_state=random_state,
        class_weight="balanced"
    )
    grid = skm.GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv_splits,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=2
    )
    
    print("Starting RF GridSearchCV…")
    grid.fit(X_tune, y_tune)
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV ROC AUC: {grid.best_score_:.4f}")
    
    return grid.best_estimator_


def evaluate_on_test(model, test_df):
    """
    Evaluate model on test_df.
    Prints classification report, confusion matrix, and ROC AUC.
    """
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"].astype(int)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n--- Test Set Evaluation ---")
    print("Classification Report:")
    print(skmtr.classification_report(y_test, y_pred))
    
    cm = skmtr.confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    auc = skmtr.roc_auc_score(y_test, y_proba)
    print(f"Test ROC AUC: {auc:.4f}")


##LOGISTIC REGRESSION 
def logreg(data):
    X = data.drop(columns=["label"])
    y = data["label"]

    # Separation
    X_train, X_test, y_train, y_test = skm.train_test_split(X, y, test_size=0.2, random_state=42)

    # model with L2-Regularisierung (Ridge)
    logreg = sklm.LogisticRegression(penalty='l2', solver='liblinear',class_weight="balanced")

    # Forward Selection
    sfs = fs.SequentialFeatureSelector(logreg,
            n_features_to_select=10,  ##(n_features = 10 is runnable, 100 is insanely slow)
            direction="forward",
            scoring='accuracy',
            cv=5)

    sfs.fit(X_train, y_train)
    print('Ausgewählte Merkmale:', sfs.feature_names_in_)

    #regulatory logistic regression (Lasso/Ridge)
    from sklearn.linear_model import LogisticRegressionCV

    # Ridge (L2) oder Lasso (L1)
    model = LogisticRegressionCV(
        Cs=10,
        cv=5,
        penalty='l1',  # oder 'l2'
        solver='liblinear',  # für L1
        scoring='accuracy',
        max_iter=1000
    )

    # training just with the featured selection
    X_train_selected = X_train[list(sfs.feature_names_in_)]
    X_test_selected = X_test[list(sfs.feature_names_in_)]

    model.fit(X_train_selected, y_train)

    #model classification
    from sklearn.metrics import classification_report, confusion_matrix

    y_pred = model.predict(X_test_selected)

    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Beste C (Regularisierungsparameter):", model.C_)

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Vorhergesagt")
    plt.ylabel("Tatsächlich")
    plt.title("Confusion Matrix")
    plt.savefig("../output/logreg_confusion_matrix")


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
            knn = skn.KNeighborsClassifier(n_neighbors=k)
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



## SVM with univariate filter

def filter(tune,k):
    y_tune=tune["label"]
    X_tune=tune.drop(columns="label")
    filter=fs.SelectKBest(k=k)
    filter.fit_transform(X_tune,y_tune)
    index=filter.get_support()
    X_filtered=X_tune.iloc[:,index]
    p_val=filter.pvalues_[index]
    variances=filter.scores_[index]/filter.scores_.sum()
    print(f"Highest P value for k = {k}:    {p_val.max()}")
    print(f"Explained Variance for k = {k}:    {variances.sum()}")
    return X_filtered, y_tune, index




def svmModel(tune):
    bestScore=-1
    train, validate = splitData(tune)
    y_validate=validate.loc[:,"label"]
    X_validate=validate.drop(columns="label")
    fits=[]
    for k in [750,1000,1250,1500,1750]:
        X,y,index = filter(train,k=k)
        model=svm.SVC(class_weight="balanced",probability=True)
        hyperparameters={

            "C" : [1,10,100,150,200],
            "kernel" : ['linear', 'poly', 'rbf', 'sigmoid']
        }
        grid = skm.GridSearchCV(
            estimator=model,
            param_grid=hyperparameters,
            scoring="f1",     # Metric for evaluation
            cv=5,                   # 5-fold cross-validation
            verbose=2               # Display the process
        )      
        grid.fit(y=y,X=X)
        X_tempValidator=X_validate.iloc[:,index]
        y_pred=grid.predict(X_tempValidator)
        score=skmtr.f1_score(y_pred=y_pred,y_true=y_validate)
        if score>bestScore:
            bestScore=score
            bestFit=grid
            bestFeaturesIndex=index
    return bestFit,bestFeaturesIndex

def testSvm(model,featuresIndex,test):
    X_test = test.drop(columns=["label"])
    X_test = X_test.iloc[:,featuresIndex]
    y_test = test["label"].astype(int)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n--- Test Set Evaluation ---")
    print(f"Best SVM model selected was: {model.best_params_} with {featuresIndex.sum()} features selected")
    print("Classification Report:")
    print(skmtr.classification_report(y_test, y_pred))
    
    cm = skmtr.confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    auc = skmtr.roc_auc_score(y_test, y_proba)
    print(f"Test ROC AUC: {auc:.4f}")


###     USING FUNCTIONS
checkna(data)
data=drop_duplicates(data)
data=maldiRename(data)
data=createDummies(data)
tune, test = splitData(data)
checkImbalance(data)
svmFit,featuresIndex = svmModel(tune)
testSvm(svmFit,featuresIndex,test)
logreg(data)

## RANDOM FOREST

best_rf = train_random_forest(tune)
evaluate_on_test(best_rf, test)

# Optional: feature importances
importances = best_rf.feature_importances_
feat_names = test.drop(columns=["label"]).columns
feat_imp_df = pd.DataFrame({
    "feature": feat_names,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\nTop 10 feature importances:")
print(feat_imp_df.head(10))


print("--- %s seconds ---" % (time.time() - start_time))
