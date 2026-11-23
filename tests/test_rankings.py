"""Regression tests: the refactored ranking math must match the original implementation."""

import importlib.util
import tkinter as tk
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "a-fine-aggregator-TOM.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("afa_tom", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _original_percentages(scores, sub_weights, stakeholder_weights):
    """Verbatim copy of the math from the original calculate_rankings."""
    scores = np.array(scores)
    sub_weights = np.array(sub_weights)
    stakeholder_weights = np.array(stakeholder_weights)

    global_criteria_weights = stakeholder_weights[:, None] * sub_weights
    mean_scores = scores.mean(axis=1)
    std_scores = np.std(scores, axis=1, ddof=0)
    std_scores_safe = np.where(std_scores == 0, 1e-6, std_scores)

    z_scores = (scores - mean_scores[:, np.newaxis, :]) / std_scores_safe[:, np.newaxis, :]
    global_criteria_weights_T = global_criteria_weights.T

    LSD = np.einsum("sak,ks->a", z_scores, global_criteria_weights_T)
    LSD_min, LSD_max = np.min(LSD), np.max(LSD)
    percentages = (LSD - LSD_min) / (LSD_max - LSD_min) if LSD_max != LSD_min else np.full_like(LSD, 0.5)
    return [int(round(p * 100)) for p in percentages]


SCORES = [
    # decision-maker 1: 3 alternatives x 4 criteria
    [[80, 20, 50, 90], [40, 70, 60, 10], [10, 90, 30, 50]],
    # decision-maker 2
    [[60, 60, 60, 60], [70, 30, 20, 80], [20, 80, 90, 40]],
    # decision-maker 3
    [[90, 10, 40, 70], [50, 50, 50, 50], [30, 60, 80, 20]],
]
SUB_WEIGHTS = [[0.25, 0.25, 0.25, 0.25], [0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]]
GLOBAL_WEIGHTS = [0.5, 0.3, 0.2]


def test_compute_rankings_matches_original_math():
    mod = _load_module()
    expected = _original_percentages(SCORES, SUB_WEIGHTS, GLOBAL_WEIGHTS)

    ranked = mod.compute_rankings(SCORES, SUB_WEIGHTS, GLOBAL_WEIGHTS)

    by_index = {r.index: r.pct for r in ranked}
    assert [by_index[i] for i in range(3)] == expected
    assert [r.pct for r in ranked] == sorted(expected, reverse=True)
    assert ranked[0].pct == 100 and ranked[-1].pct == 0


def test_ties_share_rank_and_identical_scores_give_fifty():
    mod = _load_module()
    scores = [[[50, 50], [50, 50], [50, 50]]]

    ranked = mod.compute_rankings(scores, [[0.5, 0.5]], [1.0])

    assert [r.pct for r in ranked] == [50, 50, 50]
    assert [r.rank for r in ranked] == [1, 1, 1]
    assert [r.name for r in ranked] == ["Alternative 1", "Alternative 2", "Alternative 3"]


def test_rank_numbering_skips_after_tie():
    mod = _load_module()
    # alternatives 1 and 2 identical on every criterion, 3 clearly worse
    scores = [[[90, 90], [90, 90], [10, 10]]]

    ranked = mod.compute_rankings(scores, [[0.5, 0.5]], [1.0])

    assert [(r.rank, r.pct) for r in ranked] == [(1, 100), (1, 100), (3, 0)]


def test_out_of_range_score_is_rejected():
    mod = _load_module()
    try:
        mod.compute_rankings([[[150, 0], [0, 0]]], [[0.5, 0.5]], [1.0])
    except ValueError as exc:
        assert "must be 0–100" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_equal_split_sums_to_hundred():
    mod = _load_module()
    assert mod.equal_split(1) == [100]
    assert mod.equal_split(3) == [33, 33, 34]
    assert mod.equal_split(4) == [25, 25, 25, 25]
    assert sum(mod.equal_split(7)) == 100


@pytest.fixture(scope="module")
def desktop_app():
    mod = _load_module()
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk desktop is unavailable: {exc}")
    try:
        app = mod.MCDAApp(root)
        root.update()
        yield mod, root, app
    finally:
        root.destroy()


def _widget_texts(widget):
    texts = []
    if "text" in widget.keys():
        texts.append(str(widget.cget("text")))
    for child in widget.winfo_children():
        texts.extend(_widget_texts(child))
    return texts


def test_desktop_identity_and_default_theme(desktop_app):
    mod, root, app = desktop_app
    assert root.title() == "TOMAP"
    assert app.theme.palette is mod.DARK
    assert app.theme_button.cget("text") == "Light mode"
    assert "TOMAP" not in _widget_texts(root)
    assert any(text == "Actors" for text in _widget_texts(root))
    app._toggle_theme()
    assert app.theme.palette is mod.LIGHT


def test_actor_labels_and_weight_totals_are_not_duplicated(desktop_app):
    mod, root, app = desktop_app
    app.generate_tables(2, 3, 2)
    root.update()
    texts = _widget_texts(root)
    assert [app.notebook.tab(index, "text") for index in range(2)] == ["Actor 1", "Actor 2"]
    assert not any("decision-maker" in text.lower() for text in texts)
    assert not any("Note: with only 2 alternatives" in text for text in texts)
    assert texts.count("All Actors: 100 %") == 2
    assert not any(text.startswith("Actor weights:") for text in texts)
    assert any("0 = relatively worst, 100 = relatively best" in text for text in texts)
    app.stakeholder_tables[0]["global_weight"].set(40)
    assert all(chip.cget("text") == "All Actors: 90 %" for chip in app._global_chips)
    app.calculate_rankings()
    assert "Actor weights must add up to 100 %" in app.status_var.get()
    app.reset_ratings_only()
    app.stakeholder_tables[1]["sub_weights"][0].set(0)
    app.calculate_rankings()
    assert "Actor 2" in app.status_var.get()


def test_matrix_window_fits_small_cases_and_caps_large_cases(desktop_app):
    mod, root, app = desktop_app
    factor = max(1.0, root.winfo_fpixels("1i") / 96.0)
    max_width = min(int(1120 * factor), root.winfo_screenwidth() - 40)
    max_height = min(int(760 * factor), root.winfo_screenheight() - 120)
    heights = []
    for alternatives in (2, 3):
        app.generate_tables(alternatives, 3, 2)
        root.update()
        tab = root.nametowidget(app.notebook.tabs()[0])
        assert not tab._vbar_visible
        assert not tab._hbar_visible
        assert 0 <= tab.canvas.winfo_height() - tab.inner.winfo_reqheight() <= 35
        heights.append(root.winfo_height())
    assert heights[0] < heights[1] <= max_height
    app.generate_tables(30, 10, 2)
    root.update()
    tab = root.nametowidget(app.notebook.tabs()[0])
    assert root.winfo_width() <= max_width
    assert root.winfo_height() <= max_height
    assert tab._vbar_visible and tab._hbar_visible
    app.generate_tables(2, 1, 1)
    root.update()
    assert root.winfo_height() < heights[0]


def test_results_use_relative_labels_without_clipping(desktop_app):
    mod, root, app = desktop_app
    app.generate_tables(2, 3, 1)
    app.calculate_rankings()
    root.update()
    assert "100 marks the best among alternatives and 0 the worst among alternatives." in app.results_view.card.subtitle.cget("text")
    scale = app.results_view.scale
    captions = {
        scale.itemcget(item, "text"): item
        for item in scale.find_all()
        if scale.type(item) == "text"
    }
    for caption in ("RELATIVELY WORST", "RELATIVELY BEST"):
        left, top, right, bottom = scale.bbox(captions[caption])
        assert 0 <= left < right <= scale.winfo_width()
        assert 0 <= top < bottom <= scale.winfo_height()
