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
data = data.rename(columns={"Unnamed: 0" : "type"})
M1 = np.sum(data.loc[:,"type"].str.contains("MALDI1"))
M2 = np.sum(data.loc[:,"type"].str.contains("MALDI2"))
data.loc[data.loc[:,"type"].str.contains("MALDI1"), "type"] = "M1"
data.loc[data.loc[:,"type"].str.contains("MALDI2"), "type"] = "M2"

print(f"There are: M1 {M1}, M2 {M2}")

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


##CREATE DUMMIES FROM M1 - M2 STRINGS TO 0.0 - 1.0 DUMMIES
dummies=pd.get_dummies(data["type"],drop_first=True)
dummies.reset_index(inplace=True)
data.reset_index(inplace=True)
data=data.merge(dummies,on="index",how="left")
data=data.drop(["type","index"],axis=1)
data["M2"]=data["M2"].astype(float)

###HEATMAP PLOTTER
"""
corr=data.corr()
plt.figure(figsize=(300,300))
sns.heatmap(corr)
plt.savefig("../output/heatmap")
print("plotted")
"""

###WORK IN PROGRESS

checkna(data)
data=drop_duplicates(data)

def split_data(data):
    train, test = skm.train_test_split(data, test_size=0.15)
    print("data splitted")
    print(f"train: {train.shape}\n test: {test.shape}")


