import os.path


class ModelManager():
    def save_rmse(self, models_rmses, ensemble_rmses, file_manager, run_name, artifacts_dir_path):
        iterations = []
        for i in range(len(models_rmses)):
            iteration = 'iteration_{}'.format(i)
            iterations.append(iteration)
        all_rmses = {}
        models_rmses_it = dict(zip(iterations, models_rmses))
        ensemble_rmses_it = dict(zip(iterations, ensemble_rmses))
        all_rmses['models_rmses'] = models_rmses_it
        all_rmses['ensemble_rmses'] = ensemble_rmses_it
        file_manager.save_to_json(all_rmses, os.path.join(
            artifacts_dir_path, run_name + "_rmse.json"))

    def save_mape(self, models_mapes, ensemble_mapes, file_manager, run_name, artifacts_dir_path):
        iterations = []
        for i in range(len(models_mapes)):
            iteration = 'iteration_{}'.format(i)
            iterations.append(iteration)
        all_mapes = {}
        models_mapes_it = dict(zip(iterations, models_mapes))
        ensemble_mapes_it = dict(zip(iterations, ensemble_mapes))
        all_mapes['models_mapes'] = models_mapes_it
        all_mapes['ensemble_mapes'] = ensemble_mapes_it
        file_manager.save_to_json(all_mapes, os.path.join(
            artifacts_dir_path, run_name + "_mape.json"))