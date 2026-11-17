#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE=Path(__file__).resolve().parent
DATA=HERE/'vacancy_barriers_100.csv'
if not DATA.exists(): raise SystemExit('Missing vacancy_barriers_100.csv. Run add_event_100.py first.')
df=pd.read_csv(DATA)
features=['jump_species','start_1nn_Cu','start_2nn_Cu','end_1nn_Cu','end_2nn_Cu','shared_1nn_Cu']
X=df[features]; y=df['barrier_eV'].astype(float)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.20,random_state=42,stratify=X['jump_species'])
pre=ColumnTransformer([('species',OneHotEncoder(handle_unknown='ignore',sparse_output=False),['jump_species'])],remainder='passthrough',verbose_feature_names_out=False)
rf=RandomForestRegressor(n_estimators=150,max_depth=8,min_samples_leaf=2,random_state=42,n_jobs=-1)
model=Pipeline([('preprocess',pre),('regressor',rf)])
model.fit(Xtr,ytr)
pred=model.predict(Xte)
mae=float(mean_absolute_error(yte,pred)); rmse=float(np.sqrt(mean_squared_error(yte,pred))); r2=float(r2_score(yte,pred))
model.fit(X,y)  # final model used by KMC uses all 100 labels
joblib.dump(model,HERE/'barrier_model.joblib')
metrics={'test_MAE_eV':mae,'test_RMSE_eV':rmse,'test_R2':r2,'n_events':int(len(df)),'features':features}
(HERE/'model_metrics.json').write_text(json.dumps(metrics,indent=2))
# parity
plt.figure(figsize=(5.5,5)); plt.scatter(yte,pred,s=42,alpha=.85)
lo=min(float(yte.min()),float(pred.min())); hi=max(float(yte.max()),float(pred.max()))
plt.plot([lo,hi],[lo,hi],'k--',lw=1); plt.xlabel('LAMMPS/NEB barrier (eV)'); plt.ylabel('ML-predicted barrier (eV)')
plt.title(f'Local-environment barrier model\nRMSE={rmse:.3f} eV, MAE={mae:.3f} eV')
plt.tight_layout(); plt.savefig(HERE/'barrier_parity.png',dpi=200); plt.close()
# feature importance
names=model.named_steps['preprocess'].get_feature_names_out()
imp=model.named_steps['regressor'].feature_importances_
order=np.argsort(imp)
plt.figure(figsize=(6.2,4.2)); plt.barh(np.array(names)[order],imp[order]); plt.xlabel('Random-forest feature importance'); plt.tight_layout(); plt.savefig(HERE/'feature_importance.png',dpi=200); plt.close()
print(f'Test MAE  = {mae:.4f} eV')
print(f'Test RMSE = {rmse:.4f} eV')
print(f'Test R^2  = {r2:.4f}')
print('Saved barrier_model.joblib, barrier_parity.png, and feature_importance.png')
