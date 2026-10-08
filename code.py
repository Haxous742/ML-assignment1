# %% [markdown]
# # BT2024268 – ML Assignment 1: Polynomial Regression
# Author: Vedant Mundada


# %% [code]
# Cell 1 - Imports and Configurations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold

SEED = 42
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

# %% [code]
# Cell 2 - Load Data
train1 = pd.read_csv('BT2024268_train_var1.csv')
test1  = pd.read_csv('BT2024268_test_var1.csv')
train2 = pd.read_csv('BT2024268_train_var2.csv')
test2  = pd.read_csv('BT2024268_test_var2.csv')

print("var1 train:", train1.shape, " | test:", test1.shape)
print("var2 train:", train2.shape, " | test:", test2.shape)

# %% [code]
# Cell 3 - EDA: Target Distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(train1['y'], bins=40, color='steelblue', edgecolor='white', linewidth=0.5)
axes[0].set_title('var1 – Net Power Score (y) distribution', fontsize=12, fontweight='bold')
axes[0].set_xlabel('y'); axes[0].set_ylabel('Count')
axes[0].axvline(train1['y'].mean(), color='red', linestyle='--', label=f"mean = {train1['y'].mean():.2f}")
axes[0].legend()

axes[1].hist(train2['y'], bins=40, color='darkorange', edgecolor='white', linewidth=0.5)
axes[1].set_title('var2 – Thermal Anomaly Score (y) distribution', fontsize=12, fontweight='bold')
axes[1].set_xlabel('y'); axes[1].set_ylabel('Count')
axes[1].axvline(train2['y'].mean(), color='red', linestyle='--', label=f"mean = {train2['y'].mean():.2f}")
axes[1].legend()

plt.tight_layout()
plt.savefig('fig_target_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [code]
FEAT1 = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
X1_tr = train1[FEAT1].values
y1_tr = train1['y'].values
DEGREES1 = list(range(2, 11)) # Sweeping up to 10

records1 = []
for deg in DEGREES1:
    poly = PolynomialFeatures(degree=deg, include_bias=True)
    Xp = poly.fit_transform(X1_tr)
    
    # OLS evaluation
    lr = LinearRegression()
    lr.fit(Xp, y1_tr)
    tr_r2 = r2_score(y1_tr, lr.predict(Xp))
    cv_r2_ols = cross_val_score(lr, Xp, y1_tr, cv=kf, scoring='r2')
    cv_mse_ols = -cross_val_score(lr, Xp, y1_tr, cv=kf, scoring='neg_mean_squared_error')
    
    # Ridge evaluation
    rr = Ridge(alpha=2.0)
    cv_r2_rr = cross_val_score(rr, Xp, y1_tr, cv=kf, scoring='r2')
    cv_mse_rr = -cross_val_score(rr, Xp, y1_tr, cv=kf, scoring='neg_mean_squared_error')
    
    records1.append({
        'degree': deg, 'terms': Xp.shape[1],
        'train_r2': tr_r2,
        'ols_cv_r2': cv_r2_ols.mean(), 'ols_cv_r2_std': cv_r2_ols.std(),
        'ols_cv_mse': cv_mse_ols.mean(), 'ols_cv_mse_std': cv_mse_ols.std(),
        'ridge_cv_r2': cv_r2_rr.mean(), 'ridge_cv_r2_std': cv_r2_rr.std(),
        'ridge_cv_mse': cv_mse_rr.mean(), 'ridge_cv_mse_std': cv_mse_rr.std(),
    })

df1_sweep = pd.DataFrame(records1)
print("var1 degree sweep summary:\n", df1_sweep[['degree','terms','train_r2','ols_cv_r2','ridge_cv_r2','ridge_cv_mse']])

# %% [code]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('var1 – Degree Selection (all 6 features, 5-fold CV)', fontsize=13, fontweight='bold')
degs = df1_sweep['degree']

axes[0].errorbar(degs, df1_sweep['ols_cv_r2'], yerr=df1_sweep['ols_cv_r2_std'], marker='o', label='OLS CV R²', linestyle='--', color='steelblue')
axes[0].errorbar(degs, df1_sweep['ridge_cv_r2'], yerr=df1_sweep['ridge_cv_r2_std'], marker='s', label='Ridge CV R²', linestyle='-', color='navy')
axes[0].axvline(4, color='red', linestyle='--', alpha=0.6, label='Chosen (deg=4)')
axes[0].set_ylim(-1, 1.05)
axes[0].set_xlabel('Polynomial Degree'); axes[0].set_ylabel('R²'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].errorbar(degs, df1_sweep['ols_cv_mse'], yerr=df1_sweep['ols_cv_mse_std'], marker='o', label='OLS CV MSE', linestyle='--', color='steelblue')
axes[1].errorbar(degs, df1_sweep['ridge_cv_mse'], yerr=df1_sweep['ridge_cv_mse_std'], marker='s', label='Ridge CV MSE', linestyle='-', color='navy')
axes[1].axvline(4, color='red', linestyle='--', alpha=0.6, label='Chosen (deg=4)')
axes[1].set_yscale('log') # Log scale prevents massive OLS overfitting from breaking the chart
axes[1].set_xlabel('Polynomial Degree'); axes[1].set_ylabel('MSE (Log Scale)'); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('fig_var1_degree_sweep.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [code]
poly4 = PolynomialFeatures(degree=4, include_bias=True)
X1p_tr = poly4.fit_transform(X1_tr)
X1p_te = poly4.transform(test1[FEAT1].values)

model1 = Ridge(alpha=2.0)
model1.fit(X1p_tr, y1_tr)
y1_pred_tr = model1.predict(X1p_tr)
resid1 = y1_tr - y1_pred_tr

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('var1 – Final Model Diagnostics (degree=4, Ridge α=2)', fontsize=13, fontweight='bold')

axes[0].scatter(y1_tr, y1_pred_tr, alpha=0.35, s=12, color='steelblue')
mn, mx = y1_tr.min(), y1_tr.max()
axes[0].plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Ideal')
axes[0].set_xlabel('Actual y'); axes[0].set_ylabel('Predicted y'); axes[0].set_title('Predicted vs Actual'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].scatter(y1_pred_tr, resid1, alpha=0.35, s=12, color='darkorange')
axes[1].axhline(0, color='red', linewidth=1.5, linestyle='--')
axes[1].set_xlabel('Predicted y'); axes[1].set_ylabel('Residual'); axes[1].set_title('Residuals vs Predicted'); axes[1].grid(alpha=0.3)

axes[2].hist(resid1, bins=40, color='mediumseagreen', edgecolor='white', linewidth=0.5)
axes[2].axvline(0, color='red', linestyle='--')
axes[2].set_xlabel('Residual'); axes[2].set_title('Residual Distribution'); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('fig_var1_diagnostics.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [code]
FEAT2 = ['x1', 'x2', 'x3']
X2_tr = train2[FEAT2].values
y2_tr = train2['y'].values
DEGREES2 = list(range(2, 21)) # Sweeping up to 20 

records2 = []
for deg in DEGREES2:
    poly = PolynomialFeatures(degree=deg, include_bias=True)
    Xp = poly.fit_transform(X2_tr)
    
    # Ridge evaluation
    rr = Ridge(alpha=0.05)
    cv_r2_rr = cross_val_score(rr, Xp, y2_tr, cv=kf, scoring='r2')
    cv_mse_rr = -cross_val_score(rr, Xp, y2_tr, cv=kf, scoring='neg_mean_squared_error')
    
    records2.append({
        'degree': deg, 'terms': Xp.shape[1],
        'ridge_cv_r2': cv_r2_rr.mean(), 'ridge_cv_r2_std': cv_r2_rr.std(),
        'ridge_cv_mse': cv_mse_rr.mean(), 'ridge_cv_mse_std': cv_mse_rr.std(),
    })

df2_sweep = pd.DataFrame(records2)
print("var2 degree sweep summary:\n", df2_sweep[['degree','terms','ridge_cv_r2','ridge_cv_mse']])

# %% [code]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('var2 – Degree Selection (all 3 features, 5-fold CV)', fontsize=13, fontweight='bold')
degs2 = df2_sweep['degree']

axes[0].errorbar(degs2, df2_sweep['ridge_cv_r2'], yerr=df2_sweep['ridge_cv_r2_std'], marker='s', label='Ridge CV R²', linestyle='-', color='saddlebrown')
axes[0].axvline(10, color='red', linestyle='--', alpha=0.6, label='Chosen (deg=10)')
axes[0].set_ylim(0, 1.05)
axes[0].set_xticks(range(2, 21, 2))
axes[0].set_xlabel('Polynomial Degree'); axes[0].set_ylabel('R²'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].errorbar(degs2, df2_sweep['ridge_cv_mse'], yerr=df2_sweep['ridge_cv_mse_std'], marker='s', label='Ridge CV MSE', linestyle='-', color='saddlebrown')
axes[1].axvline(10, color='red', linestyle='--', alpha=0.6, label='Chosen (deg=10)')
axes[1].set_yscale('log') # Log scale
axes[1].set_xticks(range(2, 21, 2))
axes[1].set_xlabel('Polynomial Degree'); axes[1].set_ylabel('MSE (Log Scale)'); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('fig_var2_degree_sweep.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [code]
poly10 = PolynomialFeatures(degree=10, include_bias=True)
X2p_tr = poly10.fit_transform(X2_tr)
X2p_te = poly10.transform(test2[FEAT2].values)

model2 = Ridge(alpha=0.05)
model2.fit(X2p_tr, y2_tr)
y2_pred_tr = model2.predict(X2p_tr)
resid2 = y2_tr - y2_pred_tr

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('var2 – Final Model Diagnostics (degree=10, Ridge α=0.05)', fontsize=13, fontweight='bold')

axes[0].scatter(y2_tr, y2_pred_tr, alpha=0.35, s=12, color='darkorange')
mn, mx = y2_tr.min(), y2_tr.max()
axes[0].plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='Ideal')
axes[0].set_xlabel('Actual y'); axes[0].set_ylabel('Predicted y'); axes[0].set_title('Predicted vs Actual'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].scatter(y2_pred_tr, resid2, alpha=0.35, s=12, color='steelblue')
axes[1].axhline(0, color='red', linewidth=1.5, linestyle='--')
axes[1].set_xlabel('Predicted y'); axes[1].set_ylabel('Residual'); axes[1].set_title('Residuals vs Predicted'); axes[1].grid(alpha=0.3)

axes[2].hist(resid2, bins=40, color='mediumseagreen', edgecolor='white', linewidth=0.5)
axes[2].axvline(0, color='red', linestyle='--')
axes[2].set_xlabel('Residual'); axes[2].set_title('Residual Distribution'); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('fig_var2_diagnostics.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [code]
pred1 = test1.copy()
pred1['y'] = model1.predict(X1p_te)
pred1.to_csv('BT2024268_pred_var1.csv', index=False)

pred2 = test2.copy()
pred2['y'] = model2.predict(X2p_te)
pred2.to_csv('BT2024268_pred_var2.csv', index=False)

print("\nSaved Final Prediction Files:")
print(" -> BT2024268_pred_var1.csv")
print(" -> BT2024268_pred_var2.csv")