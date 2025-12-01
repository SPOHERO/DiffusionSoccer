import os
import pickle
import importlib
from collections.abc import Mapping

def import_class(_class):
    if not isinstance(_class, str):
        return _class
    module_name = '.'.join(_class.split('.')[:-1])
    class_name = _class.split('.')[-1]
    module = importlib.import_module(module_name)
    _class = getattr(module, class_name)
    print(f"[ utils/config ] Imported {module_name}:{class_name}")
    return _class


class Config(Mapping):
    def __init__(self, _class, verbose=True, savepath=None, device=None, **kwargs):
        self._class = import_class(_class)
        self._device = device
        self._dict = kwargs

        if verbose:
            print(self)

        if savepath is not None:
            savepath = os.path.join(*savepath) if isinstance(savepath, tuple) else savepath
            pickle.dump(self, open(savepath, "wb"))
            print(f"[ utils/config ] Saved config to: {savepath}\n")

    def __repr__(self):
        string = f"\n[utils/config] Config: {self._class.__name__}\n"
        for k, v in sorted(self._dict.items()):
            string += f"    {k}: {v}\n"
        return string

    def __getitem__(self, item):
        return self._dict[item]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __call__(self, *args, **kwargs):
        instance = self._class(*args, **{**self._dict, **kwargs})
        if self._device and hasattr(instance, "to"):
            instance = instance.to(self._device)
        return instance