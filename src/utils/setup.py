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


def _as_bool(value: Any) -> bool:
    """Truthy parse for YAML bools and env strings ('1'/'true'/'yes'/'on')."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _env_override(key: str, default: Any, cast):
    """``SPARTA_<KEY>`` env var (if set) overrides ``default``, parsed by ``cast``.

    The same idiom as :func:`resolve_dataset` / :func:`resolve_experiment`, lifted so a
    single Slurm array task can pick a timing / sequence combo without editing the YAML.
    The default (already the right type from PyYAML) is returned untouched when unset.
    """
    raw = os.environ.get(f"SPARTA_{key.upper()}")
    return cast(raw) if raw is not None else default


def resolve_dynamic(data_config: Dict[str, Any]) -> bool:
    """``parameters.time_dynamic`` with ``SPARTA_TIME_DYNAMIC`` overriding the YAML.

    Lets one extraction script run both the static (``false``) and dynamic (``true``)
    passes from the same config.
    """
    return _env_override("time_dynamic", data_config['parameters']['time_dynamic'], _as_bool)


def resolve_timing(data_config: Dict[str, Any], network: str) -> Dict[str, Any]:
    """``network_construction`` timing dict, with ``SPARTA_*`` env overrides applied.

    Overridable: ``SPARTA_TIME_STEP``, ``SPARTA_TIME_WIDTH``, ``SPARTA_TIME_TYPE``,
    ``SPARTA_ECHO``, ``SPARTA_DAYS_ECHO``. Returns a shallow copy (never mutates the loaded
    config) with the same keys as ``data_config[network]['network_construction']``.
    """
    nc = dict(data_config[network]['network_construction'])
    nc['time_step'] = _env_override('time_step', nc['time_step'], int)
    nc['time_width'] = _env_override('time_width', nc['time_width'], int)
    nc['time_type'] = _env_override('time_type', nc['time_type'], str)
    nc['echo'] = _env_override('echo', nc['echo'], _as_bool)
    nc['days_echo'] = _env_override('days_echo', nc['days_echo'], int)
    return nc


def resolve_sequence(data_config: Dict[str, Any], network: str) -> Dict[str, Any]:
    """``sequence`` dict, with ``SPARTA_*`` env overrides applied.

    Overridable: ``SPARTA_K``, ``SPARTA_TASK``, ``SPARTA_N_TEST_ANCHORS``,
    ``SPARTA_N_VAL_ANCHORS``. K and task are the *free* sweep axis (no re-extraction),
    so the env override lets one array task pick them without touching the YAML.
    """
    sq = dict(data_config[network]['sequence'])
    sq['K'] = _env_override('k', sq['K'], int)
    sq['task'] = _env_override('task', sq['task'], str)
    sq['n_test_anchors'] = _env_override('n_test_anchors', sq['n_test_anchors'], int)
    sq['n_val_anchors'] = _env_override('n_val_anchors', sq['n_val_anchors'], int)
    return sq


def run_tag(timing: Dict[str, Any]) -> str:
    """Deterministic, filesystem-safe id for a timing combo — used to namespace the
    dynamic feature dir, the ``.npz`` cache, and the per-combo result/model files so a
    sweep's combos coexist instead of overwriting each other.

    ``echo`` ignores the window width in ``load_network_time`` (network_data_loader.py),
    so the tag branches on echo: echo on -> ``echo{days_echo}``, echo off ->
    ``w{time_width}``, both prefixed by the snapshot grid ``{time_type[0]}{time_step}``
    (e.g. ``d1``) so combos that share a grid — hence the same T and comparable
    K-windows — share a prefix: ``d1_echo3``, ``d1_echo7``, ``d1_w1``, ``d1_w3``.
    """
    grid = f"{str(timing['time_type'])[0]}{int(timing['time_step'])}"
    if _as_bool(timing['echo']):
        return f"{grid}_echo{int(timing['days_echo'])}"
    return f"{grid}_w{int(timing['time_width'])}"


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