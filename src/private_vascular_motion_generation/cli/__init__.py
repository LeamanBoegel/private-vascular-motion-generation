"""Study runners, verification and reporting entry points.

Ref: Sec. 4.9 (every reported cell is one arm of the same protocol).

The submodules are reached through their `__main__`-style entry points
(`python -m private_vascular_motion_generation.cli.<name>`) and through the console scripts
declared in `pyproject.toml`, so this package deliberately imports none of them: an eager
import here would re-execute the module that `runpy` is already executing.
"""
