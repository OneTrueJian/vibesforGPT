"""Latexify labels and plots"""


def latexify(sym):
    """Make a symbol use Latex"""
    if "|" in sym:
        return "|".join([latexify(s) for s in sym.split("|")])

    if sym == "G":
        sym = "\\Gamma"

    return "$\\mathrm{\\mathsf{" + str(sym) + "}}$"


def latexify_labels(labels):
    """Convert all labels to Latex"""
    return [latexify(sym) for sym in labels]
