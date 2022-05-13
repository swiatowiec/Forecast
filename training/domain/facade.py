from infrastructure.file_manager import FileManager
from domain.service import ModelService
from infrastructure.model_manager import ModelManager
from domain.dataset_factory import DatasetFactory


class TrainFacade:
    def __init__(self,
                 file_manager: FileManager,
                 service: ModelService,
                 model_manager: ModelManager):
        self._service = service
        self._file_manager = file_manager
        self._model_manager = model_manager

    def train(self,
              train_dir_path: str,
              n_splits: int,
              look_back: int,
              run_name: str,
              artifacts_dir_path: str,
              ):

        assert run_name is not None
        dataset_factory = DatasetFactory(self._file_manager)
        data_series = dataset_factory.create_from_files(train_dir_path=train_dir_path,
                                                        n_splits=n_splits,
                                                        )

        models_weights, models_preds, models_rmses, models_mapes = self._service.train(
            n_splits=n_splits,
            look_back=look_back,
            data_series=data_series,
        )
        ensemble_preds, ensemble_rmses, ensemble_mapes = self._service.train_ensemble(
            data_series=data_series,
            preds=models_preds,
            weights=models_weights,
            look_back=look_back,
        )
        self._model_manager.save_rmse(
            models_rmses=models_rmses,
            ensemble_rmses=ensemble_rmses,
            file_manager=self._file_manager,
            run_name=run_name,
            artifacts_dir_path=artifacts_dir_path,
        )
        self._model_manager.save_mape(
            models_mapes=models_mapes,
            ensemble_mapes=ensemble_mapes,
            file_manager=self._file_manager,
            run_name=run_name,
            artifacts_dir_path=artifacts_dir_path,
        )