import functools
import os
import re
from importlib import import_module
from pathlib import Path

import yaml


class ConfigLoaderMeta(type):
    def __new__(metacls, name, bases, namespace):
        cls = super().__new__(metacls, name, bases, namespace)

        cls.add_constructor("!include", cls.construct_include)

        cls.add_constructor("!path", cls.construct_path)

        cls._path_matcher = re.compile(r"\$\{([^}^{]+)\}")
        cls.add_implicit_resolver("!path", cls._path_matcher, None)

        return cls


class ConfigLoader(yaml.Loader, metaclass=ConfigLoaderMeta):
    def __init__(self, stream):
        try:
            self._root = Path(stream.name).parent
        except AttributeError:
            self._root = Path()

        super().__init__(stream)

    def construct_include(self, node):
        filename = self._root / self.construct_scalar(node)
        ext = filename.suffix.lower()

        with open(filename, "r") as f:
            if ext in (".yaml", ".yml", ".json"):
                return yaml.load(f, ConfigLoader)
            else:
                return "\n".join(
                    ln for ln in (line.strip() for line in f.read().splitlines()) if ln
                )

    def construct_path(self, node):
        def expand(match):
            expression = match.group(1)
            name, separator, default = expression.partition(":-")
            value = os.environ.get(name)
            if value is not None:
                return value
            if separator:
                return default
            raise KeyError(f"required environment variable {name!r} is not set")

        return re.sub(self._path_matcher, expand, self.construct_scalar(node))


def load_config(filename):
    load = functools.partial(yaml.load, Loader=ConfigLoader)
    with open(filename, "r") as f:
        config = load(f)

    return config


def create_config(identifier, **kwargs):

    if identifier is None:
        return None

    if isinstance(identifier, os.PathLike):
        identifier = os.fspath(identifier)
    if isinstance(identifier, str):
        if os.path.isfile(identifier):
            config = load_config(identifier)
        else:
            config = {"class_name": str(identifier), "config": {}}
    elif isinstance(identifier, dict):
        config = identifier
    else:
        raise TypeError(
            f"Expected `identifier` to be None, str or dict, found: {identifier} of type {type(identifier)}."
        )

    assert ("class_name" in config) and (
        "config" in config
    ), f"Configuration file structure error: `class_name` or `config` is not found: {config}."
    config["config"].update(kwargs)

    return config


def create_object_from_config(config, module_objects=None, partial=False):

    if config is None:
        return None

    cls_module, cls_name, cls_config = (
        config.get("module"),
        config["class_name"],
        config["config"],
    )
    if not cls_module:
        cls = module_objects[cls_name]
    else:
        cls = getattr(import_module(cls_module), cls_name)

    if not partial:
        return cls(**cls_config)
    else:
        return functools.partial(cls, **cls_config)


def create_object(identifier, module_objects=None, partial=False, **kwargs):
    if identifier is None:
        return None

    try:
        config = create_config(identifier, **kwargs)
    except TypeError:
        obj = identifier
    else:
        obj = create_object_from_config(
            config, module_objects=module_objects, partial=partial
        )

    return obj


def create_func(identifier):
    if identifier is None:
        return None

    if isinstance(identifier, os.PathLike):
        identifier = os.fspath(identifier)
    if isinstance(identifier, str):
        func_module, func_name = (
            ".".join(identifier.split(".")[:-1]),
            identifier.split(".")[-1],
        )
        func = getattr(import_module(func_module), func_name)
    else:
        func = identifier

    if callable(func):
        return func
    else:
        raise ValueError(
            f"Expected callable `func`, found: {func} of type {type(func)}."
        )
