"""Shared output-disk review for direct forecast runners and query doors."""

from __future__ import annotations

from pathlib import Path


def forecast_projection(exp, *, restart=None, io_mode="history",
                        render_products="none", stop_seconds=None) -> dict:
    """Price forecast output using configured writer variables and clocks.

    Prepared inputs and source downloads already exist when this review is
    taken. Checkpoints use the retention policy the checkpoint writer reads.
    A history-free tree still writes its configured checkpoint sets.
    """
    from woof import disk_budget
    from woof.resume import checkpoint_retention

    if io_mode not in {"history", "none"}:
        raise ValueError("io_mode must be 'history' or 'none'")
    elapsed = None
    if restart is not None:
        from woof.io.restart import _admissible_elapsed_seconds, read_restart_header

        try:
            elapsed = _admissible_elapsed_seconds(
                read_restart_header(Path(restart)).get("elapsed_seconds"),
                f"restart file {restart}")
        except (OSError, ValueError, RuntimeError, KeyError):
            # The restore owns malformed-checkpoint refusals. Until it
            # validates the checkpoint, reserve the whole forecast output.
            elapsed = None
    projection = disk_budget.projected_run_bytes(
        exp, keep_checkpoints=checkpoint_retention(), fetch=None, chain=None,
        render=io_mode == "history" and render_products not in {"", "none", None},
        render_products=render_products or "none", resume_seconds=elapsed,
        **({} if stop_seconds is None else {"stop_seconds": stop_seconds}))
    if io_mode == "none":
        projection["history_bytes"] = 0
        projection["picture_bytes"] = 0
        if "radar_bytes" in projection:
            projection["radar_bytes"] = 0
        projection["total_bytes"] = projection["checkpoint_bytes"]
        for domain in projection["domains"]:
            domain.update(history_bytes=0, history_frames=0, picture_bytes=0)
    projection["scope"] = "forecast output; existing source and prepared inputs excluded"
    return projection


def renderer_products(first_products=None, observer=None) -> str:
    """Read the product request already resolved by the rendering owner."""
    trigger = first_products or getattr(observer, "first_products", None)
    return str(getattr(trigger, "render_products", "none") or "none")


def require_output_space(exp, directory, *, restart=None, io_mode="history",
                         render_products="none", stop_seconds=None) -> dict:
    """Refuse output that would stop partway when its filesystem fills."""
    from woof import disk_budget

    projection = forecast_projection(
        exp, restart=restart, io_mode=io_mode, render_products=render_products,
        stop_seconds=stop_seconds)
    available = disk_budget.free_bytes(Path(directory))
    refusal = disk_budget.disk_refusal(projection, available)
    if refusal is not None:
        raise ValueError(refusal)
    return dict(projection, free_bytes=available)
