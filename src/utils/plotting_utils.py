"""
Plotting helpers for the maze-game analysis notebooks.

plot_heatmap                    performance across a 2-D sweep
plot_performance_by_maze_size   performance vs. maze size (one line per model)
concatenate_mazes_grid          render several mazes side by side
"""
import math
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype  # type: ignore
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t


def plot_heatmap(
    df,
    col_1='lm_1_name_pp',
    col_2='lm_2_name_pp',
    col_1_label=None,
    col_2_label=None,
    value_col='weighted_outcome',
    panel_col='maze_size',
    title_col='Some Heatmaps',
    max_panels_per_row=3,
    panel_size=4,
    panel_row_size=None,
    panel_col_size=None,
    font_size_internal=10,
    font_size_title=16,
    font_size_labels=8,
    cmap='RdBu_r',
    agg_function='mean',
    fmt='.2f',
    show_support_in_title: bool = True,
    panel_constants: Optional[dict[str, float]] = None,
    show_ci: bool = False,
    ci_multiplier: float = 1.96,
    yticklabels_rotation: float = 0,
    xticklabels_rotation: float = 35,
    show_heatbar: bool = True,
    show_panel_col_title: bool = True,
    manual_vmin: float | None = None,
    manual_vmax: float | None = None,
    figsize: tuple[float, float] | None = None
):
    """ Plot heatmaps for the given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the data to plot.
        col_1 (str): Column name for the first axis.
        col_2 (str): Column name for the second axis.
        value_col (str): Column name for the values to plot.
        panel_col (str): Column name for the panel identifier.
        title_col (str): Title for the entire figure.
        max_panels_per_row (int): Maximum number of panels per row.
        panel_size (int): Size of each panel in inches.
        panel_row_size (Optional[int]): Height of each row of panels in inches.
        panel_col_size (Optional[int]): Width of each column of panels in inches.
        font_size_internal (int): Font size for internal annotations.
        font_size_title (int): Font size for the main title.
        font_size_labels (int): Font size for axis labels.
        cmap (str): Colormap to use for the heatmaps.
        agg_function (str): Aggregation function to use for pivoting.
        fmt (str): Format string for annotations.
        show_support_in_title (bool): Whether to show average support in the title.
        panel_constants (Optional[dict[str, float]]): Dictionary of constants to subtract from each panel.
        show_ci (bool): Whether to show 95% confidence interval next to the mean.
        yticklabels_rotation (float): Rotation angle for y-axis tick labels in degrees.
                                     Use 0 for horizontal, 90 for vertical (default).
        xticklabels_rotation (float): Rotation angle for x-axis tick labels in degrees.
                                     Use 0 for horizontal, 35 for angled (default), 90 for vertical.
    """

    if show_ci and agg_function != 'mean':
        raise ValueError(
            "show_ci=True can only be used with agg_function='mean'")

    df_ = df.copy()

    # Set default axis labels if not provided
    if col_1_label is None:
        col_1_label = col_1
    if col_2_label is None:
        col_2_label = col_2

    # Get all unique maze sizes and number of subplots needed
    panels = sorted(df_[panel_col].unique())
    n = len(panels)

    # Calculate grid dimensions
    n_cols = min(n, max_panels_per_row)
    n_rows = math.ceil(n / max_panels_per_row)
    if panel_row_size is None:
        panel_row_size = panel_size
    if panel_col_size is None:
        panel_col_size = panel_size

    if figsize is None:
        figsize = (panel_col_size * n_cols, panel_row_size * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    # First, calculate all aggregated values across all panels to find the global min and max
    all_agg_values = []
    for panel in panels:
        group = df[df[panel_col] == panel]
        if not group.empty:
            pivot = group.pivot_table(
                index=col_1,
                columns=col_2,
                values=value_col,
                aggfunc=agg_function,
                observed=False
            )
            if panel_constants and panel in panel_constants:
                pivot = pivot - panel_constants[panel]
            all_agg_values.extend(pivot.values.flatten())

    # Filter out NaN values before finding min/max
    valid_agg_values = [v for v in all_agg_values if pd.notna(v)]
    vmin = min(valid_agg_values) if valid_agg_values else None
    vmax = max(valid_agg_values) if valid_agg_values else None

    if manual_vmin is not None:
        vmin = manual_vmin
    if manual_vmax is not None:
        vmax = manual_vmax

    if is_categorical_dtype(df_[col_1]):
        all_col1_values = df_[col_1].cat.categories
    else:
        all_col1_values = sorted(df_[col_1].unique())

    if is_categorical_dtype(df_[col_2]):
        all_col2_values = df_[col_2].cat.categories
    else:
        all_col2_values = sorted(df_[col_2].unique())
        all_col2_values = sorted(df_[col_2].unique())

    for idx, panel in enumerate(panels):
        row = idx // max_panels_per_row
        col = idx % max_panels_per_row

        group = df_[df_[panel_col] == panel]
        pivot = group.pivot_table(
            index=col_1,
            columns=col_2,
            values=value_col,
            aggfunc=agg_function,
            observed=False
        )

        annot_arg = True
        fmt_arg = fmt

        if show_ci:
            pivot_sem = group.pivot_table(
                index=col_1, columns=col_2, values=value_col, aggfunc='sem', observed=False)
            pivot_size = group.pivot_table(
                index=col_1, columns=col_2, values=value_col, aggfunc='size', observed=False)

            pivot_sem = pivot_sem.reindex_like(pivot)
            pivot_size = pivot_size.reindex_like(pivot)

            annot_labels = pd.DataFrame(
                index=pivot.index, columns=pivot.columns, dtype=object)
            for r in pivot.index:
                for c in pivot.columns:
                    mean_val = pivot.loc[r, c]
                    if pd.isna(mean_val):
                        annot_labels.loc[r, c] = ''
                        continue

                    # Adjust the displayed mean if a constant is provided for the panel
                    display_mean = mean_val
                    if panel_constants and panel in panel_constants:
                        display_mean = mean_val - panel_constants[panel]

                    sem_val = pivot_sem.loc[r, c]
                    size_val = pivot_size.loc[r, c]

                    label = f"{display_mean:{fmt}}"
                    if pd.notna(sem_val) and pd.notna(size_val) and size_val > 1:
                        dof = size_val - 1
                        multiplier = t.ppf(0.975, dof)
                        margin_of_error = sem_val * multiplier
                        label += f"\n(±{margin_of_error:.2f})"
                    annot_labels.loc[r, c] = label

            annot_arg = annot_labels
            fmt_arg = ''

        if panel_constants and panel in panel_constants:
            pivot = pivot - panel_constants[panel]

        panel_title = ""
        if show_panel_col_title:
            panel_title = f"{panel_col}: {panel}"
            if show_support_in_title:
                pivot_counts = group.pivot_table(
                    index=col_1,
                    columns=col_2,
                    values=value_col,
                    aggfunc='count',
                    observed=False
                )
                total_observations = int(pivot_counts.sum().sum())
                num_cells_with_data = pivot_counts.count().sum()
                avg_support = total_observations / \
                    num_cells_with_data if num_cells_with_data > 0 else 0
                panel_title += f"\n(N={total_observations}, Avg Support={avg_support:.1f})"

        # Reindex to ensure all combinations are present
        pivot = pivot.reindex(index=all_col1_values, columns=all_col2_values)
        if show_ci:
            annot_arg = annot_arg.reindex(
                index=all_col1_values, columns=all_col2_values)

        ax = axes[row, col]
        sns.heatmap(
            pivot,
            ax=ax,
            square=True,
            linewidths=1.,
            linecolor='white',
            annot=annot_arg,
            annot_kws={'size': font_size_internal},
            # Only show colorbar on the last plot
            cbar=(idx == n - 1) if show_heatbar else False,
            cbar_ax=None if idx < n -
            1 else (fig.add_axes(rect=(0.92, 0.3, 0.02, 0.4)) if show_heatbar else None),
            fmt=fmt_arg,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            mask=pivot.isna()  # Show NaN values as empty/white squares
        )

        ax.xaxis.set_ticks_position('top')

        # Only show y-labels for leftmost plots
        if col == 0:
            ax.set_yticklabels(ax.get_yticklabels(),
                               fontsize=font_size_labels,
                               rotation=yticklabels_rotation)
            ax.set_ylabel(col_1_label, fontsize=font_size_labels)
        else:
            ax.set_yticklabels([])
            ax.set_ylabel("")

        if row == 0:
            ax.set_xticklabels(
                ax.get_xticklabels(),
                rotation=xticklabels_rotation,
                ha='left',
                fontsize=font_size_labels)
            ax.set_title("")  # Remove default title
        else:
            ax.set_xticklabels([])
            ax.set_title("")

        ax.set_xlabel(col_2_label, fontsize=font_size_labels)

        ax.text(0.5, -0.15, panel_title,
                ha='center', va='top', fontsize=font_size_labels, transform=ax.transAxes)

    # Hide any unused subplots
    for idx in range(n, n_rows * n_cols):
        row = idx // max_panels_per_row
        col = idx % max_panels_per_row
        axes[row, col].set_visible(False)

    if title_col:
        plt.suptitle(title_col, fontsize=font_size_title, y=0.95)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        plt.tight_layout(rect=(0., 0.05, 0.9, 0.9))

    plt.show()

    return fig


def concatenate_mazes_grid(
    maze_strings: list[str],
    max_per_row: int = 3,
    h_separator: str = "  ",
    v_separator: str = "\n"
) -> str:
    """
    Concatenate multiple maze strings in a grid layout.

    Args:
        maze_strings: List of maze strings to concatenate
        max_per_row: Maximum number of mazes per row before wrapping
        h_separator: Horizontal separator between mazes (default: "  ")
        v_separator: Vertical separator between rows (default: newline)

    Returns:
        Grid-arranged concatenated maze string
    """
    if not maze_strings:
        return ""

    # Split into rows based on max_per_row
    rows = []
    for i in range(0, len(maze_strings), max_per_row):
        row_mazes = maze_strings[i:i + max_per_row]
        rows.append(row_mazes)

    # Process each row
    combined_rows = []
    for row_mazes in rows:
        # Split each maze in this row into lines
        maze_lines = [maze.strip().split('\n') for maze in row_mazes]

        # Get the maximum height for this row
        max_height = max(len(lines) for lines in maze_lines)

        # Get width of each maze for padding
        maze_widths = [
            max(len(line) for line in lines) if lines else 0
            for lines in maze_lines
        ]

        # Pad each maze to the same height
        for lines in maze_lines:
            while len(lines) < max_height:
                lines.append('')

        # Combine lines horizontally for this row
        row_combined = []
        for line_idx in range(max_height):
            combined_line = ""
            for maze_idx, lines in enumerate(maze_lines):
                line = lines[line_idx]
                # Pad line to maze width
                padded_line = line.ljust(maze_widths[maze_idx])

                if maze_idx > 0:
                    combined_line += h_separator
                combined_line += padded_line
            row_combined.append(combined_line)

        combined_rows.append('\n'.join(row_combined))

    # Add extra vertical separator between rows if specified
    if v_separator != "\n":
        return v_separator.join(combined_rows)
    else:
        # Add a blank line between rows
        return '\n\n'.join(combined_rows)



def plot_performance_by_maze_size(
    df: pd.DataFrame,
    model_col: str = 'model_name',
    size_col: str = 'maze_size',
    metric_col: str = 'outcome_success',
    figsize: tuple = (14, 6),
    title: str = 'Model Performance by Maze Size',
    ylabel: str = 'Success Rate',
    xlabel: str = "Model",
    marker_size: int = 8,
    ci_multiplier: float = 1.96,  # 95% CI
    group_spacing: float = 1.5,  # Space between model groups
    within_group_spacing: float = 0.5,  # Space between maze sizes within a group
    shade_alpha: float = 0.2,  # Alpha for confidence band
    line_alpha: float = 0.8,
    colors: Optional[list] = None,
    return_summary: bool = True,
    model_order: Optional[list] = None,
    y_upper: Optional[float] = None,
) -> plt.Figure | tuple[plt.Figure, pd.DataFrame]:
    """
    Plot performance with models grouped on x-axis, showing progression across maze sizes.

    Args:
        df: DataFrame with model_name, maze_size, and metric columns
        model_col: Column name for model names
        size_col: Column name for maze sizes
        metric_col: Column name for the metric to plot
        figsize: Figure size
        title: Plot title
        ylabel: Y-axis label
        marker_size: Size of markers
        ci_multiplier: Multiplier for SEM to get CI
        group_spacing: Space between model groups
        within_group_spacing: Space between points within a group
        shade_alpha: Alpha for confidence band shading
        line_alpha: Alpha for connecting lines
        colors: Optional list of colors, one per model
        return_summary: Whether to print summary statistics
        model_order: Optional order of models on the x-axis

    Returns:
        The matplotlib figure object
    """

    # Convert maze_size to numeric for sorting
    if df[size_col].dtype == 'object' and isinstance(df[size_col].iloc[0], tuple):
        df['size_numeric'] = df[size_col].apply(
            lambda x: x[0])  # Use first dimension
    else:
        df['size_numeric'] = df[size_col]

    # Calculate statistics
    stats = df.groupby([model_col, 'size_numeric'], observed=False)[metric_col].agg([
        'mean',
        'sem',
        'count'
    ]).reset_index()

    # Sort by size
    stats = stats.sort_values('size_numeric')

    # Get unique models and sizes
    if model_order is not None:
        # Use provided order, filtering to only models that exist in data
        all_models = df[model_col].unique()
        models = [m for m in model_order if m in all_models]
    else:
        models = df[model_col].unique()
    sizes = sorted(df['size_numeric'].unique())

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Generate colors for each model
    if colors is None:
        n_models = len(models)
        if n_models <= 10:
            # Use tab10 for up to 10 models
            colors = plt.cm.tab10(range(n_models))
        elif n_models <= 20:
            # Use tab20 for up to 20 models
            colors = plt.cm.tab20(range(n_models))
        else:
            # For more than 20 models, use a continuous colormap
            cmap = plt.cm.get_cmap('hsv')
            colors = [cmap(i / n_models) for i in range(n_models)]

    # Calculate x positions
    x_labels = []

    current_x = 0
    for i, model in enumerate(models):
        model_data = stats[stats[model_col] ==
                           model].sort_values('size_numeric')

        # X positions for this model's maze sizes
        x_positions = []
        means = []
        sems = []

        for j, size in enumerate(sizes):
            size_data = model_data[model_data['size_numeric'] == size]

            x_pos = current_x + (j * within_group_spacing)
            x_positions.append(x_pos)

            if not size_data.empty:
                means.append(size_data['mean'].values[0])
                sems.append(size_data['sem'].values[0])
            else:
                means.append(np.nan)
                sems.append(np.nan)

        # Plot the line connecting points with confidence band
        valid_idx = ~np.isnan(means)
        if any(valid_idx):
            valid_x = np.array(x_positions)[valid_idx]
            valid_means = np.array(means)[valid_idx]
            valid_sems = np.array(sems)[valid_idx]

            # Calculate confidence intervals
            ci_lower = valid_means - ci_multiplier * valid_sems
            ci_upper = valid_means + ci_multiplier * valid_sems

            # Use the color for this specific model (i-th model gets i-th color)
            model_color = colors[i]

            # Plot shaded confidence band
            ax.fill_between(valid_x, ci_lower, ci_upper,
                            color=model_color,
                            alpha=shade_alpha)

            # Plot line
            ax.plot(valid_x, valid_means, '-',
                    color=model_color,
                    alpha=line_alpha,
                    linewidth=2,
                    label=model)

            # Plot points
            ax.scatter(valid_x, valid_means,
                       color=model_color,
                       s=marker_size**2,
                       zorder=5,  # Make sure points are on top
                       edgecolor='white',
                       linewidth=1)

        # Store middle position for model label
        middle_x = current_x + ((len(sizes) - 1) * within_group_spacing) / 2
        x_labels.append((middle_x, model))

        # Move to the next group
        current_x += (len(sizes) - 1) * within_group_spacing + group_spacing

    # Set x-axis labels (model names)
    ax.set_xticks([x for x, _ in x_labels])
    ax.set_xticklabels([label for _, label in x_labels],
                       rotation=45, ha='right')

    # Customize plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if y_upper is not None:
        ax.set_ylim(-0.05, y_upper)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)

    # Subtle vertical lines between groups
    for i in range(1, len(models)):
        x_line = x_labels[i][0] - group_spacing/2
        ax.axvline(x_line, color='gray', linestyle=':', alpha=0.2)

    plt.tight_layout()

    # Print summary statistics
    if return_summary:
        print("\nSummary Statistics:")
        print("="*60)
        pivot = stats.pivot_table(
            values='mean',
            index=model_col,
            columns='size_numeric',
            aggfunc='first'
        ).round(3)
        print(pivot)

        return fig, pivot

    return fig
