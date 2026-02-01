from dataclasses import fields, is_dataclass, field, MISSING
from typing import Any, TypeVar
import numpy as np
import h5py
from pathlib import Path
from jax.experimental import io_callback

import jax
import jax.numpy as jnp

from jztree.data import ParticleData, FofCatalogue, squeeze_particles, squeeze_catalogue

T = TypeVar("T")

def write_to_hdf5(group: h5py.Group, data: Any, *, compression: str | None = "gzip", compression_opts: int = 4):
    """
    Write a dataclass instance into an HDF5 group.

    Rules:
      - None -> skip
      - field metadata {"static": True} -> group.attrs[name] = value
      - nested dataclass -> subgroup + recurse
      - otherwise -> dataset group[name] = array/value (overwrite if exists)
    """
    if not is_dataclass(data):
        raise TypeError("data must be a dataclass instance")

    for f in fields(data):
        name = f.name
        val = getattr(data, name)
        if val is None:
            continue

        if is_dataclass(val):
            write_to_hdf5(group.require_group(name), val, compression=compression, compression_opts=compression_opts)
            continue

        if f.metadata.get("static", False):
            group.attrs[name] = np.asarray(val) if hasattr(val, "__array__") and not isinstance(val, (str, bytes)) else val
            continue

        if name in group:
            del group[name]

        arr = np.asarray(val)
        if arr.shape == ():  # scalar
            group.create_dataset(name, data=arr.item())
        else:
            kwargs = {}
            if compression is not None:
                kwargs.update(compression=compression, compression_opts=compression_opts, shuffle=True, chunks=True)
            group.create_dataset(name, data=arr, **kwargs)

def read_from_hdf5(group: h5py.Group, cls: type[T]) -> T:
    """
    Inverse of write_to_hdf5: build `cls` (a dataclass type) from `group`.

    Rules:
      - static fields come from group.attrs[name] (if present)
      - non-static fields come from datasets group[name] (if present)
      - nested dataclasses come from subgroups group[name] (if present)
      - missing optional fields default to their dataclass defaults (or None)
      - missing required fields -> raises KeyError
    """
    if not is_dataclass(cls):
        raise TypeError("cls must be a dataclass type")

    kwargs: dict[str, Any] = {}

    for f in fields(cls):
        name = f.name
        is_static = f.metadata.get("static", False)

        # Nested dataclass field?
        nested = is_dataclass(f.type)

        if nested and name in group and isinstance(group[name], h5py.Group):
            kwargs[name] = read_from_hdf5(group[name], f.type)  # type: ignore[arg-type]
            continue

        if is_static:
            if name in group.attrs:
                val = group.attrs[name]
                # h5py may return numpy scalars/arrays; normalize scalars to Python
                if isinstance(val, np.generic):
                    val = val.item()
                kwargs[name] = val
            else:
                if f.default is not MISSING:
                    kwargs[name] = f.default
                elif f.default_factory is not MISSING:  # type: ignore[comparison-overlap]
                    kwargs[name] = f.default_factory()    # type: ignore[misc]
                else:
                    raise KeyError(f"Missing attribute '{name}' in group '{group.name}'")
            continue

        # Non-static: dataset (or subgroup if user stored nested that way)
        if name in group and isinstance(group[name], h5py.Dataset):
            kwargs[name] = group[name][()]  # numpy scalar or ndarray
            continue
        if nested and name in group and isinstance(group[name], h5py.Group):
            kwargs[name] = read_from_hdf5(group[name], f.type)  # type: ignore[arg-type]
            continue

        # Missing: use dataclass defaults if any, else error
        if f.default is not MISSING:
            kwargs[name] = f.default
        elif f.default_factory is not MISSING:  # type: ignore[comparison-overlap]
            kwargs[name] = f.default_factory()    # type: ignore[misc]
        else:
            raise KeyError(f"Missing dataset '{name}' in group '{group.name}'")

    return cls(**kwargs)

def squeeze_any(data: Any | ParticleData | FofCatalogue):
    if isinstance(data, ParticleData):
        return squeeze_particles(data)
    elif isinstance(data, FofCatalogue):
        return squeeze_catalogue(data)
    else:
        return data

def write_h5file(file_name, squeeze=True, **data_sets):
    base_dir = Path(file_name).parent
    base_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(file_name, "w") as file:
        for name, data in data_sets.items():
            if squeeze:
                data = squeeze_any(data)

            write_to_hdf5(file.require_group(name), data)

def read_h5file(file_name, **data_classes):
    base_dir = Path(file_name).parent
    base_dir.mkdir(parents=True, exist_ok=True)

    res = []

    with h5py.File(file_name, "r") as file:
        for name, cls in data_classes.items():
            res.append(read_from_hdf5(file[name], cls))
    return res

def distr_write_hdf5(base_name, squeeze=True, **kwargs):
    axis_name = jax.sharding.get_abstract_mesh().axis_names
    rank = jax.lax.axis_index(axis_name)
    def write(rank, **kwargs):
        file_name = f"{base_name}_{rank}.hdf5"
        
        print(f"Writing {file_name} on rank {rank}")
        write_h5file(file_name, squeeze=squeeze, **kwargs)
        print(f"Done writing on rank {rank}")

        return 0
    
    token = io_callback(write, jax.ShapeDtypeStruct((), jnp.int32), rank, **kwargs)
    return token