from sklearn.model_selection import TimeSeriesSplit


class Dataset:
    def __init__(self, train_df, n_splits):
        self._train_data = train_df
        self._n_splits = n_splits

    def train(self, model):
        target_col = 'depth_to_groundwater'
        X = self._train_data.loc[:, self._train_data.columns != target_col]
        X = X.rename(columns={'date': 'ds'})
        y = self._train_data.loc[:, self._train_data.columns == target_col]
        y.columns = ['y']
        cv = TimeSeriesSplit(n_splits=self._n_splits)
        for train_index, val_index in cv.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            model.train(
                X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    def get_validation_data(self, model):
        target_col = 'depth_to_groundwater'
        y = self._train_data.loc[:, self._train_data.columns == target_col]
        y.columns = ['y']
        cv = TimeSeriesSplit(n_splits=self._n_splits)
        y_vals = []
        for train_index, val_index in cv.split(y):
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            y_vals.append(y_val)
        return model.get_validation_data(y_vals=y_vals)