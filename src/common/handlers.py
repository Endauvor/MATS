import importlib
from typing import Any
from functools import wraps
from pydantic import BaseModel


def dump_model(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, BaseModel):
            return result.model_dump()
        return result
    return wrapper


def load_object(obj_path: str, default_obj_path: str = '') -> callable:
    obj_path_list = obj_path.rsplit('.', 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f'Object `{obj_name}` cannot be loaded from `{obj_path}`.')
    return getattr(module_obj, obj_name)


def load_created_object(obj_path: str, kwargs: dict) -> Any:
    class_object = load_object(obj_path)
    for field, value in kwargs.items():
        if isinstance(value, dict) and "object" in value:
            print(value["object"])
            kwargs[field] = load_created_object(value["object"], value["kwargs"])
    return class_object(**kwargs)