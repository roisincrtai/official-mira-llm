"""The MIRA lattice initializers, one per file: model/initializers/<name>.py registers <name> with
model.mira_initializer.register. Importing this package imports every such file; _common.py holds
the parts they share and registers nothing."""
import importlib
import pkgutil

for _module in pkgutil.iter_modules(__path__):
    if not _module.name.startswith("_"):
        importlib.import_module(f"{__name__}.{_module.name}")
