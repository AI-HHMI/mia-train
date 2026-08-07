"""Walk a module tree, applying a function to every submodule by qualified name.

`nn.Module.apply` passes only the module, so any initialisation that depends on *where* a module
sits -- which DINOv3's weight init needs -- cannot use it. These pass the dotted name alongside.
"""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn


def named_apply(
    fn: Callable[..., None],
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    """Call `fn(module=..., name=...)` on every submodule, mutating them in place."""
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def named_replace(
    fn: Callable[..., nn.Module],
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    """Like `named_apply`, but `fn` returns a module that takes the original's place."""
    if not depth_first and include_root:
        module = fn(module=module, name=name)
    for child_name_o, child_module in list(module.named_children()):
        child_name = ".".join((name, child_name_o)) if name else child_name_o
        new_child = named_replace(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
        setattr(module, child_name_o, new_child)

    if depth_first and include_root:
        module = fn(module=module, name=name)
    return module
