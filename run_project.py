"""SEML3211 Stock Market Analysis.
Reproduce the selected paper's Random Forest workflow and compare it with Extra Trees.
"""
from pathlib import Path
import argparse, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

warnings.filterwarnings("ignore")
ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/"results"; RESULTS.mkdir(exist_ok=True)
MODELS=RESULTS/"models"; MODELS.mkdir(exist_ok=True)
STOCKS=["SBIN.NS","HDFCBANK.NS","ICICIBANK.NS","AXISBANK.NS","KOTAKBANK.NS"]
FEATURES=["10by20","10by40","MAT20","MAT40","MSV10","MSV20","MSV40","POP9","POP19","POP39"]

def features(df):
    d=df.copy(); close=d["Close"]; volume=d["Volume"]
    d["10by20"]=close.pct_change(10)/close.pct_change(20)
    d["10by40"]=close.pct_change(10)/close.pct_change(40)
    d["MAT20"]=close/close.rolling(20).mean()-1
    d["MAT40"]=close/close.rolling(40).mean()-1
    d["MSV10"]=volume/volume.rolling(10).mean()-1
    d["MSV20"]=volume/volume.rolling(20).mean()-1
    d["MSV40"]=volume/volume.rolling(40).mean()-1
    d["POP9"]=close.pct_change(9); d["POP19"]=close.pct_change(19); d["POP39"]=close.pct_change(39)
    return d

def synthetic():
    rng=np.random.default_rng(3211); n=900; dates=pd.bdate_range("2018-01-01",periods=n)
    base=100*np.exp(np.cumsum(rng.normal(.0003,.012,n)))
    return pd.DataFrame({"Open":base,"High":base,"Low":base,"Close":base,
                         "Volume":abs(rng.normal(1e6,2e5,n))},index=dates)

def load_live():
    import yfinance as yf
    frames=[]
    benchmark=yf.download("^NSEI",start="2014-01-01",end="2022-01-01",auto_adjust=True,progress=False)["Close"]
    if isinstance(benchmark,pd.DataFrame): benchmark=benchmark.iloc[:,0]
    for ticker in STOCKS:
        raw=yf.download(ticker,start="2014-01-01",end="2022-01-01",auto_adjust=True,progress=False)
        if raw.empty: continue
        if isinstance(raw.columns,pd.MultiIndex): raw.columns=raw.columns.get_level_values(0)
        d=features(raw.dropna()); d["benchmark"]=benchmark.reindex(d.index).ffill()
        d["target"]=d["Close"].shift(-20)/d["Close"]-d["benchmark"].shift(-20)/d["benchmark"]
        d["ticker"]=ticker; frames.append(d)
    if not frames: raise RuntimeError("No live market data was downloaded.")
    return pd.concat(frames).replace([np.inf,-np.inf],np.nan).dropna(subset=FEATURES+["target"])

def prepare_demo():
    d=features(synthetic()); d["benchmark"]=d["Close"].shift(1).rolling(20).mean()
    d["target"]=d["Close"].shift(-20)/d["Close"]-d["benchmark"].shift(-20)/d["benchmark"]
    return d.replace([np.inf,-np.inf],np.nan).dropna(subset=FEATURES+["target"])

def metrics(name,y,p):
    return {"Model":name,"MAE":mean_absolute_error(y,p),"MSE":mean_squared_error(y,p),
            "RMSE":mean_squared_error(y,p)**.5,
            "MAPE":np.mean(np.abs((y-p)/np.where(np.abs(y)<1e-8,1e-8,y))),
            "R2":r2_score(y,p),"Directional_Accuracy":np.mean(np.sign(y)==np.sign(p))}

def run(d):
    X=d[FEATURES]; y=d["target"]; cut=int(len(d)*.8)
    Xtr,Xte=X.iloc[:cut],X.iloc[cut:]; ytr,yte=y.iloc[:cut],y.iloc[cut:]
    rf=RandomForestRegressor(n_estimators=300,max_features="sqrt",random_state=3211,n_jobs=-1)
    rf.fit(Xtr,ytr); prf=rf.predict(Xte)
    grid=GridSearchCV(ExtraTreesRegressor(random_state=3211,n_jobs=-1),
        {"n_estimators":[200,300],"max_features":["sqrt","log2"],"min_samples_leaf":[1,2]},
        cv=TimeSeriesSplit(n_splits=5),scoring="neg_mean_absolute_error",n_jobs=-1)
    grid.fit(Xtr,ytr); et=grid.best_estimator_; pet=et.predict(Xte)
    out=pd.DataFrame([metrics("Paper Random Forest (reproduction)",yte,prf),
                      metrics("Proposed Extra Trees",yte,pet),
                      {"Model":"Paper Random Forest (reported)","MAE":.0353,"MSE":.0034,"RMSE":.0588,"MAPE":1.5644,"R2":np.nan,"Directional_Accuracy":np.nan}])
    out.to_csv(RESULTS/"model_results.csv",index=False)
    joblib.dump(rf,MODELS/"random_forest.joblib"); joblib.dump(et,MODELS/"extra_trees.joblib")
    pd.DataFrame({"Actual":yte,"RF_Predicted":prf,"ET_Predicted":pet}).to_csv(RESULTS/"latest_predictions.csv")
    pd.DataFrame({"Feature":FEATURES,"RandomForest":rf.feature_importances_,"ExtraTrees":et.feature_importances_}).sort_values("ExtraTrees",ascending=False).to_csv(RESULTS/"feature_importance.csv",index=False)
    plt.figure(figsize=(9,5)); plt.barh(FEATURES,et.feature_importances_); plt.title("Extra Trees Feature Importance"); plt.tight_layout(); plt.savefig(RESULTS/"feature_importance.png",dpi=160); plt.close()
    plt.figure(figsize=(6,6)); plt.scatter(yte,pet,s=12,alpha=.55); plt.xlabel("Actual Excess Return"); plt.ylabel("Predicted Excess Return"); plt.title("Extra Trees: Actual vs Predicted"); plt.tight_layout(); plt.savefig(RESULTS/"prediction_scatter.png",dpi=160); plt.close()
    with open(RESULTS/"best_params.json","w") as f: json.dump(grid.best_params_,f,indent=2)
    print("\nExperiment completed.\n"); print(out.to_string(index=False)); print(f"\nResults saved in: {RESULTS}")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--demo",action="store_true"); ap.add_argument("--live",action="store_true"); a=ap.parse_args()
    if not a.demo and not a.live: ap.error("Use --demo or --live")
    run(prepare_demo() if a.demo else load_live())
