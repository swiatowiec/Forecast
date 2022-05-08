from typing import Any, List
import math
import numpy as np
import pandas as pd
from functools import lru_cache
from statistics import mean
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from pydantic import BaseModel
from sklearn.model_selection import ParameterGrid

class Iteration(BaseModel):
    trained_model: Any
    score_rmse: Any


class Model():
    def __init__(self):
        self.iterations: List[Iteration] = []
        self.n_splits = 3

    def get_iteration(self, n):
        return self.iterations[n]


class ProphetModel(Model):
    def train(self, X_train, y_train, X_val, y_val):
        model_parameters = pd.DataFrame(columns=['RMSE', 'Parameters'])
        ds = pd.DataFrame(X_train['ds'])
        ds_val = pd.DataFrame(X_val['ds'])
        train = ds.join(y_train)
        grid = self.prophet_grid_search()
        for p in grid:
            model = Prophet(changepoint_prior_scale=p['changepoint_prior_scale'],
                            n_changepoints=p['changepoint_range'],
                            seasonality_mode=p['seasonality_mode'],
                            growth=p['growth'],
                            weekly_seasonality=True,
                            daily_seasonality=True,
                            yearly_seasonality=True,
                            interval_width=0.95)
            model.fit(train)
            y_pred = model.predict(ds_val)
            score_rmse = math.sqrt(mean_squared_error(
                y_val, y_pred.tail(len(ds_val))['yhat']))
            score_mape = mean_absolute_percentage_error(
                y_val, y_pred.tail(len(ds_val))['yhat'])
            model_parameters = model_parameters.append(
                {'RMSE': score_rmse, 'MAPE': score_mape, 'Parameters': p}, ignore_index=True)
        parameters = model_parameters.sort_values(by=['RMSE'])
        parameters = parameters.reset_index(drop=True)
        best_rmse = parameters['RMSE'][0]
        best_mape = parameters['MAPE'][0]
        self.iterations.append({
            "score_rmse": best_rmse,
            "score_mape": best_mape,
            "preds": y_pred['yhat']
        })

    def prophet_grid_search(self):
        params_grid = dict()
        params_grid['growth'] = ['linear']
        params_grid['changepoint_range'] = [100, 200]
        params_grid['changepoint_prior_scale'] = [
            0.005, 0.01, 0.05, 0.5, 1, 5, 10, 20, 50, 100]
        params_grid['seasonality_mode'] = ['multiplicative', 'additive']
        grid = ParameterGrid(params_grid)
        return grid


class ArimaModel(Model):
    def train(self, X_train, y_train, X_val, y_val):
        grid = self.arima_grid_search()
        best_rmse_score, best_cfg = float("inf"), None
        for p in grid:
            order = list(p.values())
            try:
                rmse, mape, preds = self.evaluate_arima_model(
                    X_train, y_train, X_val, y_val, order)
                if rmse < best_rmse_score:
                    best_rmse_score, best_mape, best_cfg = rmse, mape, order
            except:
                continue
        self.iterations.append({
            "score_rmse": best_rmse_score,
            "score_mape": best_mape,
            "preds": preds.reset_index(drop=True)
        })

    def evaluate_arima_model(self, X_train, y_train, X_val, y_val, arima_order):
        model = ARIMA(y_train, order=arima_order)
        model_fit = model.fit()
        y_pred = model_fit.forecast(y_val.shape[0])
        rmse = math.sqrt(mean_squared_error(y_val, y_pred))
        mape = mean_absolute_percentage_error(y_val, y_pred)
        return rmse, mape, y_pred

    def arima_grid_search(self):
        params_grid = dict()
        params_grid['p_values'] = [0, 1, 2, 4, 6, 8, 10]
        params_grid['d_values'] = range(0, 3)
        params_grid['q_values'] = range(0, 3)
        grid = ParameterGrid(params_grid)
        return grid


class AutoArimaModel(Model):
    def train(self, X_train, y_train, X_val, y_val):
        ds_val = pd.DataFrame(X_val['ds'])
        auto_arima = pm.auto_arima(
            y_train,
            start_p=1,
            start_q=1,
            test='adf',
            max_p=3, max_q=3,
            m=1,
            d=None,
            seasonal=False,
            start_P=0,
            D=0,
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True)
        y_pred = auto_arima.predict(ds_val.shape[0])
        score_rmse = math.sqrt(mean_squared_error(y_val, y_pred))
        score_mape = mean_absolute_percentage_error(y_val, y_pred)
        self.iterations.append({
            "score_rmse": score_rmse,
            "score_mape": score_mape,
            "preds": pd.Series(y_pred)
        })


class LSTMModel(Model):
    def __init__(self, look_back):
        self.look_back = look_back

    def train(self, X_train, y_train, X_val, y_val):
        cfg_list = self.model_configs()
        scores = [self.repeat_evaluate(
            X_train, y_train, X_val, y_val, cfg) for cfg in cfg_list]
        scores.sort(key=lambda tup: tup[1])
        return scores

    def repeat_evaluate(self, X_train, y_train, X_val, y_val, config, n_repeats=2):
        key = str(config)
        scores = [self.walk_forward_validation(
            X_train, y_train, X_val, y_val, config) for _ in range(n_repeats)]
        result = mean(scores)
        return (key, result)

    def walk_forward_validation(self, X_train, y_train, X_val, y_val, config):
        n_input, n_nodes, n_epochs, n_batch = config
        dataset = y_train.values
        dataset_val = y_val.values

        feature_range = (-1, 0)
        scaler = MinMaxScaler(feature_range=feature_range)
        scaled_data = scaler.fit_transform(dataset)
        scaled_data_val = scaler.fit_transform(dataset_val)

        X_train, y_train = self.create_dataset(scaled_data, self.look_back)
        X_val, y_val = self.create_dataset(scaled_data_val, self.look_back)

        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))

        LSTM_model = Sequential()
        LSTM_model.add(LSTM(n_nodes, activation='relu', return_sequences=True, input_shape=(
            n_input, X_train.shape[2])))
        LSTM_model.add(Dense(n_nodes, activation='relu'))
        LSTM_model.add(Dense(1))
        LSTM_model.compile(optimizer='adam', loss='mean_squared_error')

        X_train = np.asarray(X_train).astype(np.float32)
        y_train = np.asarray(y_train).astype(np.float32)

        LSTM_model.fit(
            X_train, y_train, batch_size=n_batch, epochs=n_epochs, validation_data=(X_val, y_val))
        train_predict = LSTM_model.predict(X_train)
        valid_predict = LSTM_model.predict(X_val)

        train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
        y_train = scaler.inverse_transform([y_train])
        valid_predict = scaler.inverse_transform(valid_predict.reshape(-1, 1))
        y_val = scaler.inverse_transform([y_val])

        score_rmse = math.sqrt(mean_squared_error(
            y_val[0], valid_predict[:, 0]))
        score_mape = mean_absolute_percentage_error(
            y_val[0], valid_predict[:, 0])
        self.iterations.append({
            "score_rmse": score_rmse,
            "score_mape": score_mape,
            "preds": pd.Series(valid_predict[:, 0].flatten())
        })
        return score_rmse

    def create_dataset(self, dataset, look_back):
        X, y = [], []
        for i in range(look_back, len(dataset)):
            a = dataset[i-look_back:i, 0]
            X.append(a)
            y.append(dataset[i, 0])
        return np.array(X), np.array(y)

    def model_configs(self):
        grid = self.lstm_grid_search()
        configs = list()
        for p in grid:
            cfg = list(p.values())
            configs.append(cfg)
        return configs

    def lstm_grid_search(self):
        params_grid = dict()
        params_grid['n_input'] = [12]
        params_grid['n_nodes'] = [100]
        params_grid['n_epochs'] = [50]
        params_grid['n_batch'] = [1, 150]
        grid = ParameterGrid(params_grid)
        return grid


class MultivarianteProphetModel(Model):
    def train(self, X_train, y_train, X_val, y_val):
        date_column = ['ds']
        feature_columns = [
            'rainfall',
            'temperature',
            'drainage_volume',
            'river_hydrometry',
        ]
        features = X_train[date_column + feature_columns]
        features_val = X_val[date_column + feature_columns]
        train = features.join(y_train)
        multivar_prophet_model = Prophet()
        multivar_prophet_model.add_regressor('rainfall')
        multivar_prophet_model.add_regressor('temperature')
        multivar_prophet_model.add_regressor('drainage_volume')
        multivar_prophet_model.add_regressor('river_hydrometry')
        multivar_prophet_model.fit(train)
        y_pred = multivar_prophet_model.predict(features_val)
        score_rmse = math.sqrt(mean_squared_error(
            y_val, y_pred.tail(len(y_val))['yhat']))
        score_mape = mean_absolute_percentage_error(
            y_val, y_pred.tail(len(y_val))['yhat'])
        self.iterations.append({
            "score_rmse": score_rmse,
            "score_mape": score_mape,
            "preds": y_pred['yhat']
        })


class EnsembleWeights(Model):
    def __init__(self, models_to_ensemble, n_splits):
        self.models_to_ensemble = models_to_ensemble
        self.n_splits = n_splits

    def get_weights_and_estimators(self):
        weights = []
        all_rmses, all_preds, all_mapes = self.get_best_models(
            models=self.models_to_ensemble, n_splits=self.n_splits)
        for i in range(len(all_rmses)):
            weight = self.calculate_weights_for_ensemble(all_rmses[i])
            weights.append(weight)
        return weights, all_preds, all_rmses, all_mapes

    def get_best_models(self, models, n_splits):
        all_rmses = []
        all_preds = []
        all_mape = []
        for i in range(n_splits):
            rmses_for_iter = []
            preds_for_iter = []
            mape_for_iter = []
            for model in models:
                iteration = model.get_iteration(i)
                preds = iteration['preds']
                rmse = iteration['score_rmse']
                mape = iteration['score_mape']
                rmses_for_iter.append(rmse)
                preds_for_iter.append(preds)
                mape_for_iter.append(mape)
            all_preds.append(preds_for_iter)
            all_rmses.append(rmses_for_iter)
            all_mape.append(mape_for_iter)
        return all_rmses, all_preds, all_mape

    def calculate_weights_for_ensemble(self, rmses: list):
        """
        rmses are average rmse from every model.
        """
        rmses = self.acc_kernel(np.array(rmses))
        return rmses / np.sum(rmses)

    def acc_kernel(self, x: np.ndarray) -> np.ndarray:
        x_ref = np.min(x)
        sigma2 = x_ref / 3
        return np.exp(-0.5 * np.absolute(x - x_ref) / sigma2)


class EnsembleModel(Model):

    def __init__(self, preds, weights, look_back):
        self.look_back = look_back
        self.preds = self.adjust_prediction_length(preds)
        self.weights = weights
        self.ensemble_preds = self.calculate_ensemble_preds(
            self.weights, self.preds)

    def calculate_ensemble_preds(self, weights, preds):
        ensemble_preds = []
        for i in range(len(preds)):
            ensemble_pred_iter = []
            for j in range(len(preds[i])):
                weighted_prediction = preds[i][j] * weights[i][j]
                ensemble_pred_iter.append(weighted_prediction)
            ensemble_per_iter_sum = sum(ensemble_pred_iter)
            ensemble_preds.append(ensemble_per_iter_sum)
        return ensemble_preds

    def adjust_prediction_length(self, preds):
        for i in range(len(preds)):
            for j in range(len(preds[i])):
                if len(preds[i][j]) == 945:
                    preds[i][j] = preds[i][j][self.look_back:].reset_index(
                        drop=True)
        return preds

    def calculate_rmses_mapes(self, y_vals):
        scores_rmses = []
        scores_mapes = []
        for i in range(len(self.weights)):
            score_rmse = math.sqrt(mean_squared_error(
                y_vals[i][self.look_back:], self.ensemble_preds[i]))
            score_mape = mean_absolute_percentage_error(
                y_vals[i][self.look_back:], abs(self.ensemble_preds[i]))
            scores_rmses.append(score_rmse)
            scores_mapes.append(score_mape)
        return self.ensemble_preds, scores_rmses, scores_mapes

    def get_validation_data(self, y_vals):
        return y_vals


@ lru_cache()
def get_prophet():
    return ProphetModel()


@ lru_cache()
def get_arima():
    return ArimaModel()


@ lru_cache()
def get_auto_arima():
    return AutoArimaModel()


@ lru_cache()
def get_LSTM(look_back: int):
    return LSTMModel(look_back=look_back)


@ lru_cache()
def get_multivariante_prophet():
    return MultivarianteProphetModel()
