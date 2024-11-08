# Figures templates for Matplotlib

Templates for figure layout, styles, and axes.

The templates are designed to create consistent figure layouts with minimal visual clutter. They are entirely customizable to fit your specific document formats. The layouts help you organize visual elements on a grid, with direct control of alignments and spacing.

## Usage

```python
import numpy as np
from figure_templates import style, make_figure, set_axis

with style("slide"):
    fig = make_figure('slide_fullwidth')
    ax = fig.axes[0]

    x = np.linspace(0, 5, 100)
    ax.plot(x, np.sin(x), marker='.', markevery=15, clip_on=False)

    set_axis(ax, 'x', major_ticks=[0,5], minor_ticks=range(5))
    set_axis(ax, 'y', major_ticks=[-1, 0, 1], minor_ticks=np.arange(-1, 1, 0.5))

    ax.set_xlabel('Time')
    ax.set_ylabel('Periodic signal')
```

## Adding Custom Stylesheets

To add custom stylesheets, create a .mplstyle file and place it in one of the following directories:

- The `mpl_figure_templates/stylesheets` directory in the platform-dependent user configuration directory (following conventions of `platformdirs`).
- The `figure_templates/stylesheets` directory in the current working directory, or the `stylesheets` directory within the directory specified by the `MPL_FIGURE_TEMPLATES_CONFIG_DIR` environment variable. Environment variables can be set in a `.env` file in the current working directory.

Stylesheets are added to matplotlib's style library and can be loaded using the `with style("style_name")` context manager. Styles are named after their file name. Configuration directories are read in order of priority, with the last directory taking precedence.

Alternatively, create a custom stylesheet as a matplotlib `RcParams` object and pass it to the `style` context manager.

## Adding Custom Layouts
To add custom layouts, create a .yml file with the layout parameters and place it in one of the following directories:

- The `mpl_figure_templates/layouts` directory in the platform-dependent user configuration directory  (following conventions of `platformdirs`).
- The `figure_templates/layouts` directory in the current working directory, or the `layouts` directory within the directory specified by the `MPL_FIGURE_TEMPLATES_CONFIG_DIR` environment variable. Environment variables can be set in a `.env` file in the current working directory.

Layouts are named after their file name. Configuration directories are read in order of priority, with the last directory taking precedence.

Example of a Custom Layout YAML File :

```yaml
# Number of rows and columns in the figure
num_rows: 2
num_cols: 2

# Width and height of the figure in inches
fig_width: 8.0
fig_height: 6.0

# Base unit in points ; this unit is used for all other dimensions. If designing on a grid, this could be the size of a grid cell.
base_unit: 72

# Margins, in base units
margin_top: 0.5
margin_bottom: 0.5
margin_left: 0.5
margin_right: 0.5

# Horizontal and vertical separation between axes in base units
hsep: 0.5
vsep: 0.5

# Whether the figure is framed
is_framed: true

# Shift of the spines in base units, with respect to the axes
spine_shift: 0.2

# X-label position: a preset defined in figure_templates.axes, and padding between label and axes in base units
xlabel_preset: center
xlabel_pad: 0.5

# Y-label position: a preset defined in figure_templates.axes, and padding between label and axes in base units
ylabel_preset: top_right
ylabel_pad: 0.5
```

Alternatively, create a `figure_templates.layouts.Layout` object and use it as template argument in the `make_figure` function.

## Samples

See the `samples` directory in the GitHub repository for examples of usage, with the default templates.

The package lets control axis ticks and relocate axis labels with minimal boilerplate:

<img src="https://raw.githubusercontent.com/numerical-io/mpl_figure_templates/refs/heads/main/samples/print_aside.png" alt="Example print_aside" style="width:320px; height:auto;">

The provided templates are designed to create consistent figure layouts with minimal visual clutter.

<img src="https://raw.githubusercontent.com/numerical-io/mpl_figure_templates/refs/heads/main/samples/print_fullwidth.png" alt="Example print_fullwidth" style="width:800px; height:auto;">

A number of presets are provided to relocate x-axis and y-axis labels in various positions :

<img src="https://raw.githubusercontent.com/numerical-io/mpl_figure_templates/refs/heads/main/samples/label_positions.png" alt="Examples of label positions" style="width:1000px; height:auto;">
