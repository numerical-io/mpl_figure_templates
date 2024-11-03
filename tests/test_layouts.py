import matplotlib
import matplotlib.figure

from figure_templates.axes import AxisLabelParams
from figure_templates.layouts import Layout, make_figure

# Ensure tests use an always available, non-interactive backend
matplotlib.use("Agg")


class TestLayout:
    def should_return_parameters_as_copy(self):
        layout = Layout(
            num_rows=2,
            num_cols=3,
            fig_width=10,
            fig_height=20,
            margin_top=1,
            margin_bottom=2,
            margin_left=3,
            margin_right=4,
            hsep=5,
            vsep=6,
            is_framed=True,
            spine_shift=7,
            base_unit=8,
            xlabel_preset="dummy_x",
            xlabel_pad=9,
            ylabel_preset="dummy_y",
            ylabel_pad=10,
        )

        result = layout.asdict()

        assert result == {
            "num_rows": 2,
            "num_cols": 3,
            "fig_width": 10,
            "fig_height": 20,
            "margin_top": 1,
            "margin_bottom": 2,
            "margin_left": 3,
            "margin_right": 4,
            "hsep": 5,
            "vsep": 6,
            "is_framed": True,
            "spine_shift": 7,
            "base_unit": 8,
            "xlabel_preset": "dummy_x",
            "xlabel_pad": 9,
            "ylabel_preset": "dummy_y",
            "ylabel_pad": 10,
        }

        # Modify the result and check that the original layout is not modified
        result["num_rows"] = 33
        assert layout.num_rows == 2
        assert layout.asdict()["num_rows"] == 2


class TestMakeFigure:
    def should_make_figure(self):
        fig = make_figure("slide_square", num_rows=2, num_cols=3)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 6

    def should_let_use_name_or_object_as_label_preset(self):
        # This should work without raising an error
        fig = make_figure("slide_square", xlabel_preset="center")

        # This should work without raising an error
        dummy_label_preset = AxisLabelParams(
            verticalalignment="baseline",
            horizontalalignment="center",
            rotation="horizontal",
            rotation_mode="default",
            x=lambda pad: 2 * pad,
            y=lambda pad: 3 * pad,
            transform=lambda ax: ax.transAxes,
        )
        fig = make_figure("slide_square", xlabel_preset=dummy_label_preset)
