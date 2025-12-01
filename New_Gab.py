# =============================================================================
# SARIMA PARA PREVISÃO DE GERAÇÃO HORÁRIA DE ENERGIA FOTOVOLTAICA
# =============================================================================
# Neste bloco simulamos dados de geração horária de um sistema FV por 5 anos.
# O SARIMA é necessário porque:
#   - há forte sazonalidade DIÁRIA (sol nasce/sol se põe)
#   - e sazonalidade ANUAL (verão/inverno)
# SARIMA captura tendências + sazonalidades repetitivas.
# =============================================================================

# =============================================================================
# SARIMA PARA PREVISÃO DE GERAÇÃO HORÁRIA DE ENERGIA FOTOVOLTAICA
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ------------------------------
# Simulação de geração horária FV
# ------------------------------
dates = pd.date_range(start='2019-01-01', end='2023-12-31', freq='H')
n = len(dates)

# Sazonalidade diária (sol)
season_day = 8 * np.maximum(0, np.sin(2*np.pi * (dates.hour-6) / 24))

# Sazonalidade anual
season_year = 4 * np.sin(2*np.pi * (dates.dayofyear) / 365)

# Tendência leve
trend = np.linspace(50, 60, n)

noise = np.random.normal(0, 2, n)

generation = np.maximum(0, trend + season_day + season_year + noise)

df = pd.DataFrame({'Data': dates, 'Geracao_MWh': generation})
df.set_index('Data', inplace=True)

# ------------------------------
# Gráfico geral
# ------------------------------
plt.figure(figsize=(12, 5))
plt.plot(df.index[:500], df['Geracao_MWh'][:500])
plt.title("Geração FV - Exemplo de Aproximadamente 21 dias iniciais")
plt.ylabel("MWh")
plt.grid()
plt.show()

# ------------------------------
# Decomposição (diária)
# ------------------------------
decomp = seasonal_decompose(df['Geracao_MWh'], model='additive', period=24)
decomp.plot()
plt.show()

# ------------------------------
# ACF e PACF
# ------------------------------
fig, ax = plt.subplots(2, 1, figsize=(12, 6))
plot_acf(df['Geracao_MWh'].iloc[:2000], lags=48, ax=ax[0])
plot_pacf(df['Geracao_MWh'].iloc[:2000], lags=48, ax=ax[1])
plt.show()

# ------------------------------
# Treino/teste
# ------------------------------
train_size = int(0.9 * len(df))
train, test = df.iloc[:train_size], df.iloc[train_size:]

# ------------------------------
# SARIMA estável
# ------------------------------
order = (1, 1, 1)
seasonal_order = (1, 1, 1, 24)

model = SARIMAX(
    train['Geracao_MWh'],
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)

# Método mais estável
results = model.fit(method='nm', maxiter=80, disp=False)
print(results.summary())

# ------------------------------
# Diagnóstico
# ------------------------------
results.plot_diagnostics(figsize=(12, 8))
plt.show()

# ------------------------------
# Previsão
# ------------------------------
forecast = results.get_forecast(steps=len(test))
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()

plt.figure(figsize=(12, 5))
plt.plot(train.index[-500:], train['Geracao_MWh'][-500:], label='Treino (últimas 500h)')
plt.plot(test.index, test['Geracao_MWh'], label='Real')
plt.plot(forecast_mean.index, forecast_mean, label='Previsão', color='red')
plt.fill_between(conf_int.index,
                 conf_int.iloc[:, 0],
                 conf_int.iloc[:, 1],
                 color='pink',
                 alpha=0.3)
plt.title('Previsão da Geração Fotovoltaica – SARIMA')
plt.ylabel('MWh')
plt.grid()
plt.legend()
plt.show()

# ------------------------------
# Métricas
# ------------------------------
mae = mean_absolute_error(test['Geracao_MWh'], forecast_mean)
rmse = np.sqrt(mean_squared_error(test['Geracao_MWh'], forecast_mean))

print(f"MAE: {mae:.2f} MWh")
print(f"RMSE: {rmse:.2f} MWh")

# ------------------------------
# Previsão futura (24 horas)
# ------------------------------
future = results.get_forecast(steps=24)
future_mean = future.predicted_mean
future_conf = future.conf_int()

plt.figure(figsize=(12, 5))
plt.plot(df.index[-500:], df['Geracao_MWh'][-500:], label='Histórico (500h)')
plt.plot(future_mean.index, future_mean, label='Previsão 24h', color='red')
plt.fill_between(future_conf.index,
                 future_conf.iloc[:, 0],
                 future_conf.iloc[:, 1],
                 alpha=0.3,
                 color='pink')
plt.title('Previsão da Geração FV para as Próximas 24 Horas')
plt.grid()
plt.legend()
plt.show()