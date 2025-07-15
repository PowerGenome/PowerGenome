from pathlib import Path

from powergenome.util import load_settings

_settings = None


def initialize_settings(path: str):
    """
    Initialize the settings dictionary from a YAML file or directory.
    This should be called once at the start of your script.
    """
    global _settings
    _settings = load_settings(Path(path))


def get_settings(**kwargs):
    """
    Retrieve the settings dictionary anywhere in the package after initialization.
    Any kwargs will be added to the settings dictionary.
    Raises an error if not initialized yet.
    """
    if _settings is None and not kwargs:
        raise ValueError(
            "Settings not initialized. Call initialize_settings(path) first."
        )
    if _settings is None:
        s = {}
    else:
        s = _settings.copy()
    if kwargs:
        s.update(kwargs)
    return s
