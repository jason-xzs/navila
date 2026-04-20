"""Lightweight local fallback for s2wrapper.

This module is used only to satisfy imports when the upstream dependency
`scaling_on_scales` cannot be installed from GitHub.

The fallback keeps the same call surface used by NaVILA:
    forward(func, images, img_sizes=..., max_split_size=...)

Behavior:
- Runs a single forward pass at the current input resolution.
- Ignores multi-scale splitting arguments.
"""

from typing import Callable, Iterable, Optional


def forward(
    func: Callable,
    images,
    img_sizes: Optional[Iterable[int]] = None,
    max_split_size: Optional[int] = None,
):
    _ = img_sizes
    _ = max_split_size
    return func(images)
