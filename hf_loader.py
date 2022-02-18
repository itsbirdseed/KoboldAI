import torch
from os import path
import sys
import json
import struct
import contextlib
import itertools
import bisect
from structures import Automaton
from typing import List, Set, Optional

CHUNK_SIZE = 4096


def get_model_spec(model_type: str) -> dict:
    dict_path = path.join(path.dirname(path.realpath(__file__)), "hf-dicts", model_type + ".json")
    if not path.exists(dict_path):
        return
    with open(dict_path) as f:
        return json.load(f)


def patch_torch_load(original_torch_load, callback, model_spec: dict, n_layers: int):
    automaton: Automaton[int, dict] = Automaton()

    for k, v in model_spec["static_weights"].items():
        ks = k
        vc = v
        vc["_key"] = ks
        kb = ks.encode("utf8")
        if not automaton.add_word(struct.pack("<I", len(kb)) + kb, vc):
            raise RuntimeError(f"Invalid or duplicate model state key {repr(ks)}")
    for k, v in model_spec["layer_weights"].items():
        for layer in range(n_layers):
            ks = k.format(layer=layer)
            vc = v.copy()
            vc["_key"] = ks
            vc["_layer"] = layer
            kb = ks.encode("utf8")
            if not automaton.add_word(struct.pack("<I", len(kb)) + kb, vc):
                raise RuntimeError(f"Invalid or duplicate model state key {repr(ks)}")
    for k, v in model_spec.get("layer_weights_even", {}).items():
        for layer in range(0, n_layers, 2):
            ks = k.format(layer=layer)
            vc = v.copy()
            vc["_key"] = ks
            vc["_layer"] = layer
            kb = ks.encode("utf8")
            if not automaton.add_word(struct.pack("<I", len(kb)) + kb, vc):
                raise RuntimeError(f"Invalid or duplicate model state key {repr(ks)}")
    automaton.make_automaton()

    def patched_torch_load(f, map_location=None, **kwargs):
        weight_specs: List[dict] = []
        weight_set: Set[int] = set()
        found_set = set()
        skip_until = -1

        def callback_wrapper(storage: torch.Storage, location: str):
            print("HF", 8)
            callback_wrapper.index += 1
            print("HF", 9)
            return callback(storage, location, weight_specs[callback_wrapper.index])
        callback_wrapper.index = -1

        try:
            print("HF", 1)
            if isinstance(f, str):
                print("HF", 1, f)
                handle = open(f, "rb")
            else:
                handle = f
                position = f.tell()
                f.seek(0)
            while True:
                chunk = handle.read(CHUNK_SIZE)
                if not chunk:
                    break
                for i, v in automaton.iter(chunk):
                    print("FOUND?", i, v["_key"], id(v), v.get("_layer"))
                    if i - len(v["_key"]) < skip_until or id(v) in weight_set:
                        continue
                    print("FOUND.", i, v["_key"], id(v), v.get("_layer"))
                    weight_set.add(id(v))
                    skip_until = i
                    found_set.add(v["_key"])
                    weight_specs.append(v)
                skip_until -= CHUNK_SIZE
                if len(weight_specs) >= len(automaton):
                    break
            print("HF", 3)
            if len(automaton) != len(weight_specs):
                raise RuntimeError(f"There are {len(automaton)} weights in the model specification, but {len(weight_specs)} were found in the actual model checkpoint")
        finally:
            print("HF", 4)
            if isinstance(f, str):
                print("HF", 5, "A")
                handle.close()
                print("HF", 6, "A")
            else:
                print("HF", 5, "B")
                f.seek(position)
                print("HF", 6, "B")
        print("HF", 7)
        return original_torch_load(f, map_location=callback_wrapper, **kwargs)
    return patched_torch_load


def patch_torch_nn_parameter_new(original_torch_nn_parameter_new):
    def patched_torch_nn_parameter_new(cls, *args, **kwargs):
        patched = kwargs.pop("kai_parameter_patched", False)
        tensor = original_torch_nn_parameter_new(cls, *args, **kwargs)
        if patched:
            return tensor
        meta = tensor.to("meta")
        meta = torch.nn.Parameter(meta, kai_parameter_patched=True)
        return meta
    return patched_torch_nn_parameter_new


def get_transformers_callback(ram_blocks: int, gpu_blocks: List[int], quiet=False):
    cumulative_gpu_blocks = tuple(itertools.accumulate(gpu_blocks))
    def callback(storage: torch.Storage, location: str, spec: dict):
        print("HF", 10)
        layer: Optional[int] = spec.get("_layer")
        print("HF", 11)
        if layer is None or layer < ram_blocks:
            if not quiet:
                print(f"{spec['_key']}  ->  (CPU)  {storage.size()}", file=sys.stderr)
            return storage.to("cpu")
        device = bisect.bisect_right(cumulative_gpu_blocks, layer - ram_blocks)
        if not quiet:
            print(f"{spec['_key']}  ->  [device {device}]  {storage.size()}", file=sys.stderr)
        return storage.to(device)
    return callback


@contextlib.contextmanager
def maybe_transformers_use_hf_loader(model_type: str, n_layers: int, ram_blocks: int, gpu_blocks: List[int], enabled=True, quiet=False):
    if not enabled:
        yield False
        return
    model_spec = get_model_spec(model_type)
    if model_spec is None:
        yield False
        return
    original_torch_load = torch.load
    original_torch_nn_parameter_new = torch.nn.Parameter.__new__
    torch.load = patch_torch_load(original_torch_load, get_transformers_callback(ram_blocks, gpu_blocks), model_spec, n_layers)
    torch.nn.Parameter.__new__ = patch_torch_nn_parameter_new(original_torch_nn_parameter_new)
    yield True
    torch.load = original_torch_load
    torch.nn.Parameter.__new__ = original_torch_nn_parameter_new
