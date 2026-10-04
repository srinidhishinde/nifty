from __future__ import annotations
import math
import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score,log_loss

def wilson_interval(wins:int,samples:int,z:float=1.96)->tuple[float,float]:
    if samples<=0:return (0.0,0.0)
    p=wins/samples; den=1+z*z/samples; centre=(p+z*z/(2*samples))/den
    margin=z*math.sqrt((p*(1-p)+z*z/(4*samples))/samples)/den
    return max(0,centre-margin),min(1,centre+margin)

def classification_metrics(y_true,probabilities,classes=(-1.0,0.0,1.0)):
    if not len(y_true): return {"samples":0,"accuracy":0.0,"balanced_accuracy":0.0,"log_loss":None,"accuracy_ci_low":0.0,"accuracy_ci_high":0.0}
    actual=np.asarray(y_true); pred=np.asarray(classes)[np.argmax(probabilities,axis=1)]
    lo,hi=wilson_interval(int((pred==actual).sum()),len(actual))
    try: bal=float(balanced_accuracy_score(actual,pred))
    except ValueError: bal=0.0
    try: ll=float(log_loss(actual,probabilities,labels=list(classes)))
    except ValueError: ll=None
    return {"samples":len(actual),"accuracy":float(accuracy_score(actual,pred)),"balanced_accuracy":bal,"log_loss":ll,"accuracy_ci_low":lo,"accuracy_ci_high":hi}
