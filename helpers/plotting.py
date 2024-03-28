""" helpers for plotting """

# Tableau colors:
tableau_colors_int = [
    (0, 107, 164),
    (255, 128, 14),
    (171, 171, 171),
    (89, 89, 89),
    (95, 158, 209),
    (200, 82, 0),
    (137, 137, 137),
]

tableau_colors = [tuple(c / 255 for c in color) for color in tableau_colors_int]
