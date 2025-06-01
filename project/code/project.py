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
import sklearn.pipeline as skp

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

def splitData(data, test_size):
    tune, test = skm.train_test_split(data, test_size=test_size)
    tune.reset_index(inplace=True)
    tune.drop(axis=1, inplace=True,columns="index")
    test.reset_index(inplace=True)
    test.drop(axis=1, inplace=True,columns="index")
    print("data splitted")
    print(f"tune: {tune.shape}\n test: {test.shape}")
    return tune, test

def checkImbalance(data):
    print(f"Data is unbalanced: {data["label"].sum(axis=0)/data.shape[0]} resistant bacteria sample and {1-data["label"].sum(axis=0)/data.shape[0]} non resistant bacteria samples")



##LOGISTIC REGRESSION 
def logreg(data):
    X = data.drop(columns=["label"])
    y = data["label"]

    n_features=10

    # Separation
    X_train, X_test, y_train, y_test = skm.train_test_split(X, y, test_size=0.2, random_state=42)

    # model with L2-Regularisierung (Ridge)
    logreg = sklm.LogisticRegression(penalty='l2', solver='liblinear',class_weight="balanced")

    # Forward Selection
    sfs = fs.SequentialFeatureSelector(logreg,
            n_features_to_select=n_features,  ##(n_features = 10 is runnable, 100 is insanely slow, ideal would be 500?) RUNNING OVERNIGHT WITH 100
            direction="forward",
            scoring='accuracy',
            cv=5,
            n_jobs=-1)

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
        max_iter=1000,
        n_jobs=-1
    )

    # training just with the featured selection
    X_train_selected = X_train[list(sfs.feature_names_in_)]
    X_test_selected = X_test[list(sfs.feature_names_in_)]

    model.fit(X_train_selected, y_train)
    return model

## SVM with univariate filter


def svmModel(tune):

    y=tune.loc[:,"label"]
    X=tune.drop(columns="label")

    pipe = skp.Pipeline([
        ("filter", fs.SelectKBest()),
        ("SVM", svm.SVC(class_weight="balanced", probability=True))
    ])

    hyperparameters={
        "filter__k" : [500, 1000, 1250, 1500, 1750],
        "SVM__C" : [10,100,125,150,175],
        "SVM__kernel" : ['linear', 'poly', 'rbf', 'sigmoid']
    }
    grid = skm.GridSearchCV(
        estimator=pipe,
        param_grid=hyperparameters,
        scoring="f1",     
        cv=5,                 
        verbose=2,
        n_jobs=-1             

    )      
    grid.fit(y=y,X=X)
    bestFit = grid.best_estimator_
    bestHyperparameters = grid.best_params_
    print(f"The best hyperparameters selected were:   {bestHyperparameters}")

    return bestFit

def testSvm(model,featuresIndex,test):
    X_test = test.drop(columns=["label"])
    X_test = X_test.iloc[:,featuresIndex]
    y_test = test["label"].astype(int)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print("\nSVM Evaluation:")
    print(f"Best SVM model selected was: {model.best_params_} with {featuresIndex.sum()} features selected")
    cm = skmtr.confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy:   {skmtr.accuracy_score(y_true=y_test,y_pred=y_pred)}")
    print(f"Precision:   {skmtr.precision_score(y_true=y_test,y_pred=y_pred)}")
    print(f"Recall:   {skmtr.recall_score(y_true=y_test,y_pred=y_pred)}")
    print(f"F1:   {skmtr.f1_score(y_true=y_test,y_pred=y_pred)}")
    auc = skmtr.roc_auc_score(y_test, y_proba)
    print(f"Test ROC AUC:   {auc}")
    plotConfusionMatrix(y_test,y_pred,"SVM")

    ###       NAMDOEL

def train_random_forest(tune_df, cv_splits=5, random_state=2025):

    # Split X / y
    X_tune = tune_df.drop(columns=["label"])
    y_tune = tune_df["label"].astype(int)
    
    # Parameter grid
    param_grid = {
        "n_estimators": [300, 350, 400, 450, 500],
        "max_depth": [None, 10, 20 ],
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
    print(f"Best Hyperparameters: {grid.best_params_}")
    print(f"Best CV ROC AUC: {grid.best_score_:.4f}")
    
    return grid.best_estimator_


def evaluate_on_test(model, test_df,name):
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"].astype(int)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{name} Evaluation:")
    cm = skmtr.confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy:   {skmtr.accuracy_score(y_true=y_test,y_pred=y_pred)}")
    print(f"Precision:   {skmtr.precision_score(y_true=y_test,y_pred=y_pred)}")
    print(f"Recall:   {skmtr.recall_score(y_true=y_test,y_pred=y_pred)}")
    print(f"F1:   {skmtr.f1_score(y_true=y_test,y_pred=y_pred)}")
    auc = skmtr.roc_auc_score(y_test, y_proba)
    
    print(f"Test ROC AUC: {auc:.4f}")

    plotConfusionMatrix(y_test,y_pred,name)
    
def plotConfusionMatrix(y_test,y_pred,name):
    cm = skmtr.confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Value")
    plt.ylabel("Actual Value")
    plt.title("Confusion Matrix")
    plt.savefig(f"../output/confusion_matrix_{name}")
    print(f"Plotted {name}")
        ###        NAMDOEL

def correlationFilter(tune,threshold):
    X=tune.drop(columns="label")
    y_var=tune.loc[:,"label"]
    toRemove=[]
    CorrelationMatrix=X.corr().abs()
    print("Correlation Matrix Done")
    for y in range(0,6000):
        for x in range(y+1,6000):
            if y!=x and CorrelationMatrix.iloc[y,x]>=threshold:
                toRemove.append(X.columns[y])
    toRemove=pd.Series(toRemove)
    toRemove.drop_duplicates(inplace=True)
    filtered_X=X.drop(columns=toRemove)
    return filtered_X,y_var


def knn(X,y):
    model=skn.KNeighborsClassifier()
    hyperparameters={

        "n_neighbors" : [5,10,50,150,300],
        "algorithm"   : ["auto", "ball_tree", "kd_tree", "brute"]

    }

    grid = skm.GridSearchCV(
        estimator=model,
        param_grid=hyperparameters,
        scoring="roc_auc",     
        cv=5,                 
        verbose=2,
        n_jobs=-1             

    )      
    grid.fit(y=y,X=X)
    print(f"Best hyperparameters selected: {grid.best_params_}")
    return grid


###     USING FUNCTIONS

checkna(data)
data=drop_duplicates(data)
data=maldiRename(data)
data=createDummies(data)
tune, test = splitData(data,0.15)
checkImbalance(data)

knn_X,knn_y = correlationFilter(tune,0.8)
knnFit = knn(knn_X,knn_y)
test_knn=test.loc[:,list(knn_X.columns[:]) +["label"]]
evaluate_on_test(model=knnFit,test_df=test_knn,name="KNN")


svmFit = svmModel(tune)
evaluate_on_test(svmFit,test,"SVM")


LogRegModel=logreg(data)
evaluate_on_test(LogRegModel,test,"Logistic_Regression")

## RANDOM FOREST         NAMDOEL

best_rf = train_random_forest(tune)
evaluate_on_test(best_rf, test,"Random_Forest")

# Optional: feature importances
importances = best_rf.feature_importances_
feat_names = test.drop(columns=["label"]).columns
feat_imp_df = pd.DataFrame({
    "feature": feat_names,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\nTop 10 feature importances:")
print(feat_imp_df.head(10))

## RANDOM FOREST         NAMDOEL


print("--- %s seconds ---" % (time.time() - start_time))
