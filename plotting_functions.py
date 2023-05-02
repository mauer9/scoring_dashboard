import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html
import numpy as np
import pandas as pd
import plotly.io as pio

from sklearn.metrics import roc_curve, roc_auc_score
pio.templates.default = "plotly_dark"


def no_figure_plot():
    return go.Figure().add_annotation(x=2, y=2, text="No Data",
                                      font=dict(family="sans serif", size=25, color="crimson"),
                                      showarrow=False, yshift=10)


def psi_single_variable_plot(psi_var_table):
    # psi_table = monitoring.psi_variable_table(style=style)
    fig = go.Figure()
    if style == 'detailed':
        var_table = psi_var_table[psi_var_table['Variable'] == var]
        fig.add_trace(go.Bar(x=var_table.index, y=var_table['Count A (%)']))
        fig.add_trace(go.Bar(x=var_table.index, y=var_table['Count E (%)']))
    return fig


def psi_variable_plot(psi_variable_table):
    if psi_variable_table is None:
        return no_figure_plot()
    # CHECK STYLE ONLY SUMMARY OR DETAILED
    fig = go.Figure()
    # if style == 'summary':
    fig.add_trace(go.Bar(x=psi_variable_table['Variable'], y=psi_variable_table['PSI']))
    # elif style == 'detailed':
    #     fig = go.Figure()
    #     fig
    return fig


### TAKE DATA FROM monitoring.psi_table(). same data as here
def psi_plot_ly(psi_table):
    if psi_table is None:
        return no_figure_plot()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=psi_table.index, y=psi_table['Count A (%)']))
    fig.add_trace(go.Bar(x=psi_table.index, y=psi_table['Count E (%)']))
    return fig


# def psi_plot_ly(psi_table):
#     fig = go.Figure()
#     fig.add_trace(go.Bar(x=psi_table.index, y=psi_table['Count A (%)']))
#     fig.add_trace(go.Bar(x=psi_table.index, y=psi_table['Count E (%)']))
#     return fig


def plot_ks(y, y_pred, title=None, xlabel=None, ylabel=None):
    if y is None or y_pred is None:
        return go.Figure().add_annotation(x=2, y=2, text="No Data to Display",
                                          font=dict(family="sans serif", size=25, color="crimson"),
                                          showarrow=False, yshift=10)


    import plotly.express as px

    # pio.templates.default = "plotly_dark"

    y = pd.Series(y)
    y_pred = pd.Series(y_pred)
    n_samples = y.shape[0]
    # n_samples = len(y)
    n_event = np.sum(y)
    n_nonevent = n_samples - n_event

    idx = y_pred.argsort()
    yy = y[idx]
    pp = y_pred[idx]

    cum_event = np.cumsum(yy)
    cum_population = np.arange(0, n_samples)
    cum_nonevent = cum_population - cum_event

    p_event = cum_event / n_event
    p_nonevent = cum_nonevent / n_nonevent

    p_diff = p_nonevent - p_event

    ks_score = np.max(p_diff)
    ks_max_idx = np.argmax(p_diff)
    # Define the plot settings
    print('plot')
    print(p_diff)
    if title is None:
        title = "Kolmogorov-Smirnov"
    if xlabel is None:
        xlabel = "Threshold"
    if ylabel is None:
        ylabel = "Cumulative probability"

    # plt.title(title, fontdict={'fontsize': 14})
    # plt.xlabel(xlabel, fontdict={'fontsize': 12})
    # plt.ylabel(ylabel, fontdict={'fontsize': 12})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pp, y=p_event))
    fig.add_trace(go.Scatter(x=pp, y=p_nonevent))
    return fig


def plot_auc_roc(y, y_pred, title=None, xlabel=None, ylabel=None):
    # savefig = False, fname = None, ** kwargs)
    if y is None or y_pred is None:
        return no_figure_plot()




    print('THIS IS Y: ', y[0:10])
    print('THIS IS Y_PRED: ', y_pred[0:10])
    fpr, tpr, _ = roc_curve(y, y_pred)
    auc_roc = roc_auc_score(y, y_pred)

    # # Define the plot settings
    # if title is None:
    #     title = "ROC curve"
    # if xlabel is None:
    #     xlabel = "False Positive Rate"
    # if ylabel is None:
    #     ylabel = "True Positive Rate"

    layout = dict(
        title='ROC-AUC',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend_title='Legend Title',
        font=dict(
            family='Courier New, monospace',
            size=18,
            color='red'
        )
    )

    fig = go.Figure(layout=layout)
    fig.add_trace(go.Scatter(x=fpr, y=fpr, name='Stupid model'))
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name='Your model'))


    return fig


def plot_hist_dist(factor1, factor2):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=factor1, histnorm='probability density'))
    fig.add_trace(go.Histogram(x=factor2, histnorm='probability density', opacity=0.5))

    fig.update_layout(barmode='overlay')
    # Reduce opacity to see both histograms
    # fig.update_traces(opacity=0.3)
    return fig


# Plots feature importances
def plot_feature_importances(feat_imp_df):
    fig = px.bar(feat_imp_df.sort_values(['Importance']), x="Importance", y="Feature", orientation='h')
    fig.update_layout(xaxis=dict(rangeslider=dict(visible=True),
                                 type="linear"))
    return fig


# Creates card with information on the dashboard
def create_card(value, card_id, title, description):
    return dbc.Card(
        dbc.CardBody(
            [
                html.H4(title, id=f"{card_id}-title"),
                html.H2(f'{value:.3f}', id=f"{card_id}-value"),
                html.P(description, id=f"{card_id}-description")
            ]
        ),
        body=True,
        color='dark',
        outline=True
    )


def new_create_card(title, card_id, new_value, prev_value, description=None):
    def f(v):
        if v is None:
            return v
        elif v == 'N/A':
            return v
        return f"{v:.3f}"

    card = dbc.Card(
        dbc.CardBody(
            [
                html.H4(title, id=f"{card_id}-title"),
                html.H2(f(new_value), id=f"{card_id}-value"),
                html.H6(f'Initial: {f(prev_value)}', id=f"{card_id}-prev-value"),
                html.P(description, id=f"{card_id}-description")
            ]
        ),
        body=True,
        color='dark',
        outline=True
    )
    return card