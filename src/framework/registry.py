from __future__ import annotations

from typing import Dict, Type
from .base import ModelInterface, CalibratorInterface, OptimizerInterface, EvaluatorInterface


_MODELS: Dict[str, Type[ModelInterface]] = {}
_CALIBRATORS: Dict[str, Type[CalibratorInterface]] = {}


def register_model(model_cls: Type[ModelInterface]):
    key = getattr(model_cls, 'id', model_cls.__name__)
    _MODELS[key] = model_cls
    return model_cls


def get_model(key: str) -> Type[ModelInterface]:
    return _MODELS[key]


def list_models():
    return list(_MODELS.keys())


def register_calibrator(calib_cls: Type[CalibratorInterface]):
    key = getattr(calib_cls, 'method_name', calib_cls.__name__)
    _CALIBRATORS[key] = calib_cls
    return calib_cls


def get_calibrator(key: str) -> Type[CalibratorInterface]:
    return _CALIBRATORS[key]


def list_calibrators():
    return list(_CALIBRATORS.keys())
