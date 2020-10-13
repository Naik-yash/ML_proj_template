import pandas as pd 
from sklearn import model_selection

if __name__=="__main__":
    train= pd.read_csv("input/train.csv")
    train.kfold= -1

    train=train.sample(frac=1).reset_index(drop=True)

    kf= model_selection.StratifiedKFold(n_splits=5, shuffle=False, random_state=50)

#target in train data set is not known, refer -https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/data?select=train.csv for more details

    for fold, (train_idx, val_idx) in enumerate (kf.split(X=train, y=train.target.values)):
        print(len(train_idx),len(val_idx))
        train.loc[val_idx,'kfold']=fold

    train.to_csv("input/train_folds.csv", index=False)



