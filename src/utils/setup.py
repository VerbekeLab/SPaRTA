# LOAD MODULES
import os
from typing import Any, Dict

# Third party
import yaml

# CUSTOM FUNCTIONS
def load_config(file_path: str) -> Dict[str, Any]:
    """
    Load a configuration file from a file path.

    Parameters:
    file_path (str): The file path of the configuration file.

    Returns:
    dict: The loaded configuration as a dictionary.
    """
    with open(file_path) as file:
        config = yaml.safe_load(file)

    return config


def resolve_dataset(data_config: Dict[str, Any]) -> str:
    """
    Active dataset, with the env var ``SPARTA_DATASET`` overriding the YAML.

    Lets a single Slurm array task pick its dataset without editing
    config/data/config.yaml. Falls back to ``parameters.dataset`` when unset.
    """
    return os.environ.get("SPARTA_DATASET", data_config['parameters']['dataset'])


def resolve_experiment(method_config: Dict[str, Any]) -> str:
    """
    Active experiment, with the env var ``SPARTA_EXPERIMENT`` overriding the YAML.

    Falls back to the top-level ``experiment`` key when unset.
    """
    return os.environ.get("SPARTA_EXPERIMENT", method_config['experiment'])


def suggest_param(trial, name: str, spec: Any):
    """
    Map a YAML search-space entry to an Optuna ``trial.suggest_*`` call.

    Spec forms (kept deliberately small so the config stays readable):

    - ``list``  -> ``suggest_categorical`` (e.g. ``[1e-5, 1e-4, 1e-3]``)
    - ``{low, high, step?}``              -> ``suggest_int`` (default)
    - ``{low, high, step?, log?, type: float}`` -> ``suggest_float``
    - any scalar -> returned as-is (a fixed, untuned value)

    ``trial`` is an Optuna ``Trial``; this helper never imports optuna itself.
    """
    if isinstance(spec, list):
        return trial.suggest_categorical(name, spec)
    if isinstance(spec, dict):
        low, high = spec['low'], spec['high']
        if spec.get('type') == 'float':
            return trial.suggest_float(name, low, high,
                                       step=spec.get('step'), log=spec.get('log', False))
        return trial.suggest_int(name, low, high, step=spec.get('step', 1))
    return spec