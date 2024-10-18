from functools import partial
import importlib
import os
import shutil
from torch.utils.data import DataLoader
from typing import Any, Dict, Type
import yaml


class DotConfig:
    """Helper class to allow "." access to dictionaries."""

    def __init__(self, cfg):
        self._cfg = cfg

    def __getattr__(self, k) -> Any:
        v = self._cfg[k]
        if isinstance(v, dict):
            return DotConfig(v)
        return v

    def __getitem__(self, k) -> Any:
        return self.__getattr__(k)

    def __contains__(self, k) -> bool:
        try:
            v = self._cfg[k]
            return True
        except KeyError:
            return False

    def to_dict(self):
        return self._cfg


def load_yaml(yaml_path: str) -> DotConfig:
    """Loads a YAML configuration file."""
    with open(yaml_path, "r") as fp:
        return DotConfig(yaml.load(fp, yaml.CLoader))


def instantiate_from_config(config, use_config_struct: bool = False) -> Any:
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    if use_config_struct:
        return get_obj_from_str(config["target"])(config["params"])
    else:
        return get_obj_from_str(config["target"])(**config.get("params", dict()))


def instantiate_partial_from_config(config, use_config_struct: bool = False) -> Any:
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    if use_config_struct:
        return partial(get_obj_from_str(config["target"]), config["params"])
    else:
        return partial(
            get_obj_from_str(config["target"]), **config.get("params", dict())
        )


def type_from_config(config) -> Type:
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])


def kwargs_from_config(config) -> Dict:
    if not "params" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return config.get("params", dict())


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def cycle(dataloader: DataLoader):
    """Cycles through the dataloader class forever.

    Useful for when you want to cycle through a DataLoader for
    a finite number of timesteps.
    """
    while True:
        for data in dataloader:
            yield data


def copy_to_drive(input_path: str, drive_folder_name: str):
    """Copies a file to google drive in colab.

    Args:
        input_path: The full path to the file to copy
        drive_folder_name: The folder path in Drive to copy to.
            This path will exist under 'My Drive', and should be
            formatted like: 'Training/Checkpoints'.
    """
    try:
        from google.colab import drive
    except ImportError:
        print(f"Must be runing in Google Colab, copying disabled.")
        return

    drive_mount_path = "/content/drive"
    drive.mount(drive_mount_path, force_remount=True)

    output_file_path = os.path.join(drive_mount_path, "MyDrive", drive_folder_name)

    # Make sure the output directory exists
    os.makedirs(output_file_path, exist_ok=True)
    shutil.copy(input_path, output_file_path)
