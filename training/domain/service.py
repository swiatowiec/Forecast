from domain.dataset import Dataset
from domain.model_repository import get_prophet, get_arima, get_auto_arima, get_LSTM, get_multivariante_prophet
from domain.model_repository import EnsembleModel, EnsembleWeights


class ModelService:
    def train(self,
              n_splits,
              look_back,
              data_series: Dataset):

        prophet = get_prophet()
        data_series.train(prophet)

        arima = get_arima()
        data_series.train(arima)

        auto_arima = get_auto_arima()
        data_series.train(auto_arima)

        lstm = get_LSTM(look_back)
        data_series.train(lstm)

        multivar_prophet = get_multivariante_prophet()
        data_series.train(multivar_prophet)

        ensemble = EnsembleWeights(
            models_to_ensemble=[prophet, arima, auto_arima, lstm, multivar_prophet], n_splits=n_splits)
        weights, preds, rmses, mapes = ensemble.get_weights_and_estimators()
        return weights, preds, rmses, mapes

    def train_ensemble(self, data_series: Dataset, preds, weights, look_back):
        ensemble = EnsembleModel(
            preds, weights, look_back)
        y_vals = data_series.get_validation_data(ensemble)
        ensemble_preds, rmses, mapes = ensemble.calculate_rmses_mapes(y_vals)
        return ensemble_preds, rmses, mapes