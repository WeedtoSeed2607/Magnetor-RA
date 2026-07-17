"""Concrete acquisition sources, one module per upstream.

Each source implements :class:`magnetor.interfaces.DomainSource` for a single
domain and is the *only* place that touches an external system.
"""
