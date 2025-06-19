import os
import pandas as pd

# Automatically locate the CSV relative to this file
csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'jmolcolors.csv')
df_colors = pd.read_csv(csv_path)

def get_color(atom):
    """
    Returns RGB color tuple corresponding to atom symbol using jmolcolors.csv
    """
    try:
        r = df_colors[df_colors['atom'] == atom]['R'].values[0] / 255.0
        g = df_colors[df_colors['atom'] == atom]['G'].values[0] / 255.0
        b = df_colors[df_colors['atom'] == atom]['B'].values[0] / 255.0
        return (r, g, b)
    except IndexError:
        return (0.5, 0.5, 0.5)
