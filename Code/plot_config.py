"""
Font and style configuration for thesis plots.
Import this module to ensure consistent styling across all figures.
"""

import matplotlib.pyplot as plt

# Font sizes
FONTSIZE_TITLE = 16
FONTSIZE_LABEL = 14
FONTSIZE_LEGEND = 12
FONTSIZE_TICK = 14

# Figure sizes (width, height)
FIGSIZE_SINGLE_COL = (8, 6)
FIGSIZE_DOUBLE_COL = (14, 6)
FIGSIZE_SQUARE = (8, 8)

# Font settings
FONT_FAMILY = 'sans-serif'
FONT_SERIF = 'Times New Roman'

# Apply global font settings
def set_thesis_style():
    """Apply consistent font and style settings for all plots."""
    plt.rcParams['font.family'] = FONT_FAMILY
    plt.rcParams['font.size'] = FONTSIZE_LABEL
    plt.rcParams['axes.titlesize'] = FONTSIZE_TITLE
    plt.rcParams['axes.labelsize'] = FONTSIZE_LABEL
    plt.rcParams['xtick.labelsize'] = FONTSIZE_TICK
    plt.rcParams['ytick.labelsize'] = FONTSIZE_TICK
    plt.rcParams['legend.fontsize'] = FONTSIZE_LEGEND
    plt.rcParams['figure.titlesize'] = FONTSIZE_TITLE
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['grid.alpha'] = 0.3
