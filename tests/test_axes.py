import matplotlib
import matplotlib.pyplot as plt

from figure_templates.axes import AxisLabelParams

# Ensure tests use an always available, non-interactive backend
matplotlib.use("Agg")


class TestLabelPreset:
    def should_return_parameters(self):
        preset = AxisLabelParams(
            verticalalignment="baseline",
            horizontalalignment="center",
            rotation="horizontal",
            rotation_mode="default",
            x=lambda pad: 2 * pad,
            y=lambda pad: 3 * pad,
            transform=lambda ax: ax.transAxes,
        )

        fig, ax = plt.subplots()
        label_pad = 1
        result = preset(ax, label_pad)

        # Check the transform
        assert result["transform"] == ax.transAxes

        # Check all other parameters
        result.pop("transform")
        assert result == {
            "verticalalignment": "baseline",
            "horizontalalignment": "center",
            "rotation": "horizontal",
            "rotation_mode": "default",
            "x": 2,
            "y": 3,
        }
