import plotly.express as px
import plotly.graph_objects as go

import numpy as np
import pandas as pd

def plot_GP(gp, tr, n_pixels=500):
    _n_estimators = np.linspace(tr._lb[0], tr._ub[0], n_pixels)
    _learning_rates = np.logspace(tr._lb[1], tr._ub[1], n_pixels)

    _X = np.column_stack(
        [
            np.tile(_n_estimators, n_pixels),
            np.repeat(np.log10(_learning_rates), n_pixels),
        ]
    )
    _y = gp.predict(_X)

    return go.Heatmap(
        x=_n_estimators,
        y=_learning_rates,
        z=_y.reshape(n_pixels, n_pixels),
        opacity=1.0,
        colorscale="RdBu",
    )


def plot_cost(cost, tr, n_pixels=500):
    _n_estimators = np.linspace(tr._lb[0], tr._ub[0], n_pixels)
    _learning_rates = np.logspace(tr._lb[1], tr._ub[1], n_pixels)

    _X = np.column_stack(
        [
            np.tile(_n_estimators, n_pixels),
            np.repeat(np.log10(_learning_rates), n_pixels),
        ]
    )
    _y = cost(_X)

    return go.Heatmap(
        x=_n_estimators,
        y=_learning_rates,
        z=_y.reshape(n_pixels, n_pixels),
        opacity=1.0,
        colorscale="RdBu",
    )


def plot_suggestions(suggestions, visible_to_opt):
    sugg_df = pd.DataFrame(suggestions)
    template = (
        "<b>id:%{customdata[0]:d}</b><br>"
        "n estimstors:%{x:d}<br>"
        "learning rate:%{y:.3f}<br>"
        "model value: %{customdata[1]:.3f}"
    )
    return go.Scatter(
        x=sugg_df.n_estimators,
        y=sugg_df.learning_rate,
        mode="text",
        customdata=np.column_stack([sugg_df.index, visible_to_opt]),
        hovertemplate=template,
        text=[f"{idx+1}" for idx in sugg_df.index],
        textposition="middle center",
    )