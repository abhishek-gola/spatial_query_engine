"""Blind, click-to-label annotation UI for benchmark items.

Deliberately separate from `sqe.viewer`. The viewer's job is to *explain* a
resolution -- it calls the resolver, shows the answer, the runner-up and the
ambiguity flags. Every one of those is a blindness leak at annotation time, and
the cheapest way to guarantee they never appear is for the annotation server to
have no code path that can produce them. Nothing in `serve()` constructs a
`Resolver`.
"""

from .server import serve

__all__ = ["serve"]
