"""Content fingerprints independent of ndarray strides and object alias layout."""
import hashlib
import json

import numpy as np
from sklearn.base import BaseEstimator

from stprotein2.data import ReferenceTransform
from stprotein2.graph import FrozenRouter, digest_array


def canonical(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays need an explicit identity schema")
        return {"ndarray": digest_array(value)}
    if isinstance(value, np.generic):
        return canonical(value.item())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return {type(value).__name__: [canonical(item) for item in value]}
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("Identity dictionaries require string keys")
        return {key: canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (ReferenceTransform, FrozenRouter, BaseEstimator)):
        cls = type(value)
        return {"class": cls.__module__ + "." + cls.__qualname__, "state": canonical(vars(value))}
    raise TypeError("Unsupported identity value: " + type(value).__name__)


def content_identity(value):
    encoded = json.dumps(canonical(value), sort_keys=True, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()
