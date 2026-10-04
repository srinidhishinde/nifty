from __future__ import annotations
from dataclasses import dataclass
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

@dataclass
class HorizonModel:
    horizon_minutes:int
    feature_columns:tuple[str,...]
    selected_columns:tuple[str,...]
    classes:tuple[float,...]
    estimator:object
    model_version:str

def fit_model(X,y,*,feature_columns,horizon_minutes,l1_c,l2_regularization,random_state,model_version):
    selector=SelectFromModel(LogisticRegression(penalty="l1",C=l1_c,solver="saga",max_iter=1500,random_state=random_state),threshold="median")
    base=Pipeline([("scale",StandardScaler()),("l1_select",selector),("gb",HistGradientBoostingClassifier(max_iter=160,learning_rate=0.06,max_leaf_nodes=15,l2_regularization=l2_regularization,random_state=random_state))])
    estimator=CalibratedClassifierCV(base,method="sigmoid",cv=3); estimator.fit(X,y)
    return HorizonModel(horizon_minutes,tuple(feature_columns),tuple(feature_columns),(-1.0,0.0,1.0),estimator,model_version)

def save_model(model,path): joblib.dump(model,path)
def load_model(path): return joblib.load(path)
