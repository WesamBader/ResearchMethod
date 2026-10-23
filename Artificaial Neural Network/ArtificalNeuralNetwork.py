from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = pd.DataFrame({
    'Compound': ['Methane','Water','Propane','Ethanol','Formic Acid','Acetic Acid','Butane','Acetone','Benzene','Toluene','Octane'],
    'MW': [16,18,44,46,46,60,58,58,78,92,114],
    'BoilingPoint': [-161,100,-42,78,101,118,-1,56,80,111,125]
})
DATA.to_csv(ROOT / 'chemistry_dataset.csv', index=False)

# Scale both inputs and targets to make neural network optimization stable.
# max_iter=5000 and tol=0 with n_iter_no_change > max_iter prevents early stopping.
def train_network(X, y, layers=(10,10)):
    xs = StandardScaler().fit(X)
    ys = StandardScaler().fit(y.reshape(-1,1))
    model = MLPRegressor(hidden_layer_sizes=layers, activation='relu', solver='adam',
                         learning_rate_init=0.005, max_iter=5000, tol=0.0,
                         n_iter_no_change=5001, early_stopping=False, random_state=42)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, message='.*Maximum iterations.*')
        model.fit(xs.transform(X), ys.transform(y.reshape(-1,1)).ravel())
    def predict(values):
        values = np.asarray(values, dtype=float).reshape(-1,1)
        return ys.inverse_transform(model.predict(xs.transform(values)).reshape(-1,1)).ravel()
    return model, predict

def scores(y_true, y_pred):
    return {'MAE': mean_absolute_error(y_true,y_pred),
            'MSE': mean_squared_error(y_true,y_pred),
            'R2': r2_score(y_true,y_pred)}

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(ROOT / filename, dpi=180)
    print('Saved:', filename)
    plt.show()  # Close the graph window to proceed.
    plt.close()

X = DATA[['MW']].to_numpy(dtype=float)
y = DATA['BoilingPoint'].to_numpy(dtype=float)
linear = LinearRegression().fit(X,y)
linear_pred = linear.predict(X)
base_net, base_predict = train_network(X,y)
net_pred = base_predict(X)

print('\nCHAPTER 13: ARTIFICIAL NEURAL NETWORKS')
print(DATA.to_string(index=False))
print('\nTRAINING DATA METRICS (not unseen test performance)')
print(pd.DataFrame({'Linear Regression': scores(y,linear_pred),
                    'Neural Network (10,10)': scores(y,net_pred)}).round(3))
print('Neural network training iterations:', base_net.n_iter_)

xgrid = np.linspace(X.min()-5, X.max()+5, 300).reshape(-1,1)
plt.figure(figsize=(8,5))
plt.scatter(X[:,0], y, label='Actual compounds')
plt.plot(xgrid,linear.predict(xgrid),label='Linear regression')
plt.xlabel('Molecular weight (g/mol)');plt.ylabel('Boiling point (°C)')
plt.title('Linear Regression: Boiling Point vs Molecular Weight');plt.legend()
save_plot('linear_regression_fit.png')

plt.figure(figsize=(8,5))
plt.scatter(X[:,0],y-linear_pred)
plt.axhline(0,linestyle='--')
plt.xlabel('Molecular weight (g/mol)');plt.ylabel('Residual (°C)')
plt.title('Linear Regression Residuals')
save_plot('linear_regression_residuals.png')

# Hold out Butane and retrain both models without it.
mask = DATA['Compound'] != 'Butane'
train_X = X[mask];train_y = y[mask]
butane_X = X[~mask];butane_y = y[~mask][0]
held_linear = LinearRegression().fit(train_X,train_y)
held_net, held_predict = train_network(train_X,train_y)
butane_linear = float(held_linear.predict(butane_X)[0])
butane_neural = float(held_predict(butane_X)[0])
print('\nBUTANE HELD-OUT TEST (Butane excluded from training)')
print(f'Actual: {butane_y:.2f} °C')
print(f'Linear prediction: {butane_linear:.2f} °C | absolute error: {abs(butane_linear-butane_y):.2f} °C')
print(f'Neural prediction: {butane_neural:.2f} °C | absolute error: {abs(butane_neural-butane_y):.2f} °C')
plt.figure(figsize=(8,5))
plt.bar(['Actual','Linear','Neural network'],[butane_y,butane_linear,butane_neural])
plt.ylabel('Boiling point (°C)');plt.title('Held-Out Butane Predictions')
save_plot('butane_prediction_comparison.png')

plt.figure(figsize=(8,5))
plt.scatter(X[:,0],y,label='Actual compounds')
plt.plot(xgrid,linear.predict(xgrid),label='Linear regression')
plt.plot(xgrid,base_predict(xgrid),label='Neural network (10,10)')
plt.xlabel('Molecular weight (g/mol)');plt.ylabel('Boiling point (°C)')
plt.title('Linear Regression vs Neural Network');plt.legend()
save_plot('model_comparison.png')

# ReLU of raw molecular weight: all inputs are positive.
weights = np.linspace(-10,120,300)
plt.figure(figsize=(8,5))
plt.plot(weights,np.maximum(0,weights),label='ReLU(x) = max(0, x)')
plt.scatter(X[:,0],np.maximum(0,X[:,0]),label='Compound MW values')
plt.xlabel('Input molecular weight');plt.ylabel('ReLU output')
plt.title('ReLU Applied to Molecular Weight');plt.legend()
save_plot('relu_activation.png')

print('\nARCHITECTURE EXPERIMENTS')
print('Choose 1–4 hidden layers and 1–10 neurons per layer.')
print('Press Enter without typing anything to finish.')
while True:
    answer = input('\nNumber of hidden layers (1–4, Enter to quit): ').strip()
    if not answer:
        break
    if not answer.isdigit() or not 1 <= int(answer) <= 4:
        print('Please enter a whole number from 1 to 4.')
        continue
    layers = []
    for i in range(int(answer)):
        while True:
            value = input(f'Neurons in hidden layer {i+1} (1–10): ').strip()
            if value.isdigit() and 1 <= int(value) <= 10:
                layers.append(int(value))
                break
            print('Please enter a whole number from 1 to 10.')
    architecture = tuple(layers)
    custom_net, custom_predict = train_network(X,y,architecture)
    custom_pred = custom_predict(X)
    print('\nArchitecture:', architecture)
    print('Training iterations:',custom_net.n_iter_)
    print('Training metrics:', {k:round(v,3) for k,v in scores(y,custom_pred).items()})
    plt.figure(figsize=(8,5))
    plt.scatter(X[:,0],y,label='Actual compounds')
    plt.plot(xgrid,base_predict(xgrid),label='Original network (10,10)')
    plt.plot(xgrid,custom_predict(xgrid),label=f'Custom network {architecture}')
    plt.xlabel('Molecular weight (g/mol)');plt.ylabel('Boiling point (°C)')
    plt.title('Neural Network Architecture Comparison');plt.legend()
    name = 'architecture_'+'_'.join(map(str,architecture))+'.png'
    save_plot(name)
print('\nFinished. All plots are saved in:', ROOT)
