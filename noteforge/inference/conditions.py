"""Render optional density/style requests into the frozen model's input tensors.

Condition encoders and the density grid are part of the published model contract.
Exposure construction, condition dropout and target-derived supervision are private.
"""
from noteforge.inference.density import grid_density_curve
from noteforge.inference.controls import encode_controls


def condition_inputs(mode, request, *, bands=None, segments=(), render_cell_us=100_000, scale=16.):
    """Controls plus a caller-supplied density curve; nothing reads a target chart."""
    controls = encode_controls(mode, dict(bands or {}), (int(request.start_us), int(request.end_us)),
                               origin_us=int(request.start_us))
    curve = grid_density_curve(list(segments), request.start_us, request.end_us,
                               cell_us=int(render_cell_us), scale=scale)
    return dict(controls=controls, **curve)


def unconditional_inputs(mode, request, *, render_cell_us=100_000, scale=16.):
    return condition_inputs(mode, request, render_cell_us=render_cell_us, scale=scale)
