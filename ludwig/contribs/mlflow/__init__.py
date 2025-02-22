import logging
import os
from typing import Any, Dict

from ludwig.callbacks import Callback
from ludwig.constants import TRAINER
from ludwig.data.dataset.base import Dataset
from ludwig.globals import MODEL_HYPERPARAMETERS_FILE_NAME, TRAIN_SET_METADATA_FILE_NAME
from ludwig.utils.data_utils import chunk_dict, flatten_dict, save_json, to_json_dict
from ludwig.utils.package_utils import LazyLoader

mlflow = LazyLoader("mlflow", globals(), "mlflow")

logger = logging.getLogger(__name__)


def _get_or_create_experiment_id(experiment_name):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(name=experiment_name)


class MlflowCallback(Callback):
    def __init__(self, tracking_uri=None):
        self.experiment_id = None
        self.run = None
        self.run_ended = False
        self.tracking_uri = tracking_uri
        self.training_set_metadata = None
        self.config = None
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

    def on_preprocess_end(
        self, training_set: Dataset, validation_set: Dataset, test_set: Dataset, training_set_metadata: Dict[str, Any]
    ):
        self.training_set_metadata = training_set_metadata

    def on_hyperopt_init(self, experiment_name):
        self.experiment_id = _get_or_create_experiment_id(experiment_name)

    def on_hyperopt_trial_start(self, parameters):
        # Filter out mlflow params like tracking URI, experiment ID, etc.
        params = {k: v for k, v in parameters.items() if k != "mlflow"}
        self._log_params({"hparam": params})

    def on_train_init(self, base_config, experiment_name, output_directory, **kwargs):
        # Experiment may already have been set during hyperopt init, in
        # which case we don't want to create a new experiment / run, as
        # this should be handled by the executor.
        if self.experiment_id is None:
            mlflow.end_run()
            self.experiment_id = _get_or_create_experiment_id(experiment_name)
            run_name = os.path.basename(output_directory)
            self.run = mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name)

        mlflow.log_dict(to_json_dict(base_config), "config.yaml")

    def on_train_start(self, config, **kwargs):
        self.config = config
        self._log_params({TRAINER: config[TRAINER]})

    def on_train_end(self, output_directory):
        _log_artifacts(output_directory)
        if self.run is not None:
            mlflow.end_run()
            self.run_ended = True

    def on_trainer_train_setup(self, trainer, save_path, is_coordinator):
        if not is_coordinator:
            return

        # When running on a remote worker, the model metadata files will only have been
        # saved to the driver process, so re-save it here before uploading.
        training_set_metadata_path = os.path.join(save_path, TRAIN_SET_METADATA_FILE_NAME)
        if not os.path.exists(training_set_metadata_path):
            save_json(training_set_metadata_path, self.training_set_metadata)

        model_hyperparameters_path = os.path.join(save_path, MODEL_HYPERPARAMETERS_FILE_NAME)
        if not os.path.exists(model_hyperparameters_path):
            save_json(model_hyperparameters_path, self.config)

    def on_eval_end(self, trainer, progress_tracker, save_path):
        mlflow.log_metrics(progress_tracker.log_metrics(), step=progress_tracker.steps)

        _log_model(save_path)

    def on_epoch_end(self, trainer, progress_tracker, save_path):
        mlflow.log_metrics(progress_tracker.log_metrics(), step=progress_tracker.steps)

        _log_model(save_path)

    def on_trainer_train_teardown(self, trainer, progress_tracker, is_coordinator):
        if self.run is not None:
            mlflow.end_run()

    def on_visualize_figure(self, fig):
        # TODO: need to also include a filename for this figure
        # mlflow.log_figure(fig)
        pass

    def prepare_ray_tune(self, train_fn, tune_config, tune_callbacks):
        from ray.tune.integration.mlflow import mlflow_mixin

        return mlflow_mixin(train_fn), {
            **tune_config,
            "mlflow": {
                "experiment_id": self.experiment_id,
                "tracking_uri": mlflow.get_tracking_uri(),
            },
        }

    def _log_params(self, params):
        flat_params = flatten_dict(params)
        for chunk in chunk_dict(flat_params, chunk_size=100):
            mlflow.log_params(chunk)

    def __setstate__(self, d):
        self.__dict__ = d
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        if self.run and not self.run_ended:
            mlflow.end_run()
            self.run = mlflow.start_run(run_id=self.run.info.run_id, experiment_id=self.run.info.experiment_id)


def _log_artifacts(output_directory):
    for fname in os.listdir(output_directory):
        lpath = os.path.join(output_directory, fname)
        if fname == "model":
            _log_model(lpath)
        else:
            mlflow.log_artifact(lpath)


def _log_model(lpath):
    # Lazy import to avoid requiring this package
    from ludwig.contribs.mlflow.model import log_saved_model

    log_saved_model(lpath)
