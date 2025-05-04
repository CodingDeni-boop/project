features=pd.read_csv("../../DRIAMS-EC/driams_Escherichia_coli_Ceftriaxone_features.csv")
labels=pd.read_csv("../../DRIAMS-EC/driams_Escherichia_coli_Ceftriaxone_labels.csv")
data=features.merge(labels)

THESE PATHS MUST BE EDITED FOR YOUR DIRECTORY.
