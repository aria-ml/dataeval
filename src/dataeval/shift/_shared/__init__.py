"""Shared scoring backends for drift and OOD detectors.

Each module implements the core math for one detection strategy
(k-neighbors, reconstruction, domain classifier) and is consumed
by the corresponding Drift* and OOD* classes in sibling packages.
"""
