# Adapted from https://github.com/tvst/st-annotated-text

import streamlit.components.v1

from htbuilder import HtmlElement, div, span, styles
from htbuilder.units import px, rem, em


import numpy as np
from matplotlib.cm import get_cmap
import streamlit as st


def contrastColor(R, G, B, luminance_switch=0.60):
    """
    Build a contrastive color for background to foreground
    https://stackoverflow.com/a/1855903/249341
    """

    # Counting the perceptive luminance - human eye favors green color...
    luminance = (0.299 * R + 0.587 * G + 0.114 * B) / 255.0

    if luminance > luminance_switch:
        return (0, 0, 0)  # bright colors - black font
    else:
        return (255, 255, 255)


def annotation(body, background, color, **style):
    """Build an HtmlElement span object with the given body and annotation label. """

    if "font_family" not in style:
        style["font_family"] = "sans-serif"

    main_style = styles(background=background, color=color,)

    return span(style=main_style)(body)

    # return span(style=main_style)(body, span(style=substyle) )


def annotated_text(
    df,
    text_column="text",
    value_column="vals",
    cmap_name="Blues",
    scale_values=True,
    **kwargs
):
    """Writes test with annotations into your Streamlit app.

    Parameters
    ----------

    *df : pandas DataFrame
    """

    style = styles(font_family="sans-serif", line_height="1.5", font_size=px(32),)

    out = div(_class="", style=style)
    cmap = get_cmap(cmap_name)

    for text, val in zip(df[text_column], df[value_column]):

        if scale_values:
            val = (val + 0.25) ** 2.5

        color = (np.array(cmap(val)[:3]) * 255).astype(np.uint8)
        contrast = contrastColor(*color)

        hex_color = "#%02x%02x%02x" % (color[0], color[1], color[2])
        contrast_color = "#%02x%02x%02x" % (contrast[0], contrast[1], contrast[2])

        out(annotation(body=text + " ", color=contrast_color, background=hex_color))

        # out(annotation(body=text, color=contrast_color, background=hex_color))
        # out(annotation(body=' '))

    streamlit.components.v1.html(str(out), **kwargs)
