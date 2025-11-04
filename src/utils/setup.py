# LOAD MODULES
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