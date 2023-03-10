import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# from xgboost import XGBClassifier
# from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from typing import Tuple


class Mushrooms:

    DATA = "./mushroom.csv"
    TARGET = "class"

    def import_data(self) -> pd.DataFrame:
        return pd.read_csv(self.DATA)

    def preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        for i in df:
            uniq = df[i].unique()
            new = []
            for j in df[i]:
                new.append(np.where(uniq == j)[0][0])
            df[i] = new
        return df

    def train_test_shape(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
    ):
        nl = "\n"
        print(
            f"X_train: {X_train.shape}{nl} X_test: {X_test.shape}{nl} \
    y_train: {y_train.shape}{nl} y_test: {y_test.shape}"
        )

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame]:
        y = df[self.TARGET]
        X = df.loc[:, df.columns != self.TARGET]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33, random_state=42
        )
        return X_train, X_test, y_train, y_test


prepro = Mushrooms()
df = prepro.preprocessing(prepro.import_data())
# X_train, X_test, y_train, y_test = prepro.split(df)
# prepro.train_test_shape(X_train, X_test, y_train, y_test)


class MushroomsClassif(Mushrooms):
    X_train, X_test, y_train, y_test = prepro.split(df)

    #   def aa(self):
    #     print(self.X_train)

    # dd = MushroomsClassif()
    # dd.aa()

    def logcl(self):
        clf = LogisticRegression(random_state=24, solver="lbfgs", max_iter=10).fit(
            self.X_train, self.y_train
        )
        return clf

    def cv(self, model):
        scores = cross_val_score(model, self.X_train, self.y_train, cv=5)
        print(scores)

    def metrics(self, model):

        f1 = f1_score(self.y_test, model.predict(self.X_test))
        roc_auc = roc_auc_score(self.y_test, model.predict_proba(self.X_test)[:, 1])
        preds = model.predict(self.X_test)

        nl = "\n"
        return f"Metrics:{nl} F1: {f1}{nl} ROC AUC:{roc_auc}{nl} accur:{accuracy_score(self.y_test, preds)}"


pred = MushroomsClassif()
model = pred.logcl()
# pred.cv(model = m)

print(f"{pred.metrics(model=model)}")
