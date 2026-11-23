from __future__ import annotations
import numpy as np

def engagement_weight(likes,comments,shares,cw=2.0,sw=3.0):
    x=np.nan_to_num(likes)+cw*np.nan_to_num(comments)+sw*np.nan_to_num(shares)
    return 1.0+np.log1p(np.maximum(x,0))

def pool_matrix(x:np.ndarray,weights=None,use_max=False,use_min=False)->dict[str,np.ndarray]:
    if x.ndim!=2: raise ValueError("Expected [items, dimensions]")
    if len(x)==0:return {}
    out={"mean":x.mean(0),"std":x.std(0)}
    if weights is not None:
        w=np.asarray(weights,float); w=np.where(np.isfinite(w)&(w>0),w,1.0)
        out["weighted_mean"]=(x*w[:,None]).sum(0)/w.sum()
    if use_max: out["max"]=x.max(0)
    if use_min: out["min"]=x.min(0)
    return out
