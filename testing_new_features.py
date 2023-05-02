######################### TO-DO #########################
# 1. User-ID based sessions?
# 2. Redis database when in prod
# 3. Add dark/light themes
# 4. Add all models status page (JSON to store all model metrics on GitLab)

#########################################################

from dash import Dash, dcc, html, Input, Output, State
from flask_caching import Cache
import dash_bootstrap_components as dbc
# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Input, Output, State
from dash import Dash, dcc, html, Input, Output, ctx
import dash_bootstrap_components as dbc
import plotly.express as px

from plotting_functions import plot_hist_dist, plot_feature_importances, create_card, psi_plot_ly, plot_auc_roc, \
    plot_ks, psi_variable_plot, new_create_card
from helper_functions import col_dropper, fix_dtypes, DataFrameImputer, AutoPrepareScoreCard
# import plotly.graph_objects as go
import pandas as pd
# from datetime import date
import sqlite3
from sklearn.metrics import roc_curve, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
import json
import helper_functions
import joblib

from optbinning import BinningProcess
from optbinning.scorecard import ScorecardMonitoring
import logging
import sys
import io
from contextlib import redirect_stdout
import time
from docxtpl import DocxTemplate, InlineImage

import credit_py_validation as cpv

from plotly.tools import mpl_to_plotly

from page_content import generate_search_bar, generate_navbar, generate_sidebar, generate_footer
from styles import get_content_style, get_footer_style, get_sidebar_style, get_sidebar_hidden_style, get_modal_style
import pages.accuracy
import pages.stability

import logging

logger = logging.getLogger(__name__)

class DashLogger(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream=stream)
        self.logs = list()

    def emit(self, record):
        try:
            msg = self.format(record)
            self.logs.append(msg)
            self.logs = self.logs[-1000:]
            self.flush()
        except Exception:
            self.handleError(record)

dash_logger = DashLogger(stream=sys.stdout)
logger.addHandler(dash_logger)

# Global variables
HALYK_LOGO = 'assets/halyk_logo.png'
# PRIVATE_TOKEN = ''
model_list = helper_functions.get_all_models()

app = Dash(external_stylesheets=[dbc.themes.CYBORG])

navbar = generate_navbar(model_list, HALYK_LOGO)

# the style arguments for the sidebar. We use position:fixed and a fixed width
CONTENT_STYLE, CONTENT_STYLE1 = get_content_style()
FOOTER_STYLE = get_footer_style()
SIDEBAR_STYLE = get_sidebar_style()
SIDEBAR_HIDDEN = get_sidebar_hidden_style()
MODAL_STYLE = get_modal_style()

sidebar = generate_sidebar(SIDEBAR_STYLE)

content = html.Div(
    id="page-content",
    style=CONTENT_STYLE)

footer = generate_footer(FOOTER_STYLE)

generate_button = html.Div(
    html.Button('Generate report',
                id='generate-report-button',
                n_clicks=0))

MODAL_CONTENT = {
    "margin": "90px",
    "padding": "30px",
    "background-color": "white",
    'textAlign': 'center'
}

modal = html.Div([
    html.Div([
        html.Div([
            'Please, choose a model from the dropdown on top, retard',
        ]),

        html.Hr(),
        html.Button('Close', id='modal-close-button')
    ],
        style=MODAL_CONTENT,
        className='modal-content',
    ),
],
    id='modal',
    className='modal',
    style=MODAL_STYLE
)

app.layout = html.Div(
    [
        dcc.Store(id='intermediate-value'),
        dcc.Store(id='side_click'),
        dcc.Location(id="url"),
        navbar,
        sidebar,
        modal,

        dcc.Loading(
            id='loading-1',
            type='default',
            # fullscreen=True,
            # children=html.Div(id='loading-output-1'),
            # children=['intermediate-value', content]
            children=content
        ),
        # content,

        dcc.Interval(
            id='log-update2',
            interval=1 * 1000  # in milliseconds
        ),
        html.Div(id='log2'),

        generate_button,
        footer
    ]
)

@app.callback(
    Output('log2', 'children'),
    [Input('log-update2', 'n_intervals')])
def update_logs(interval):
    return [html.Div(log) for log in dash_logger.logs]


# @app.callback(Output('my-div', 'children'), [Input('button', 'n_clicks')])
# def add_log(click):
#     logger.warning("Important Message")





@app.callback(Output('modal', 'style'),
              [Input('modal-close-button', 'n_clicks')])
def close_modal(n):
    if n is None:
        return get_modal_style()
    elif (n is not None) and (n > 0):
        return {"display": "none"}


@app.callback(
    [
        Output("sidebar", "style"),
        Output("page-content", "style"),
        Output("side_click", "data"),
    ],

    [Input("btn_sidebar", "n_clicks")],
    [
        State("side_click", "data"),
    ]
)
def toggle_sidebar(n, nclick):
    if n:
        if nclick == "SHOW":
            sidebar_style = SIDEBAR_HIDDEN
            content_style = CONTENT_STYLE1
            cur_nclick = "HIDDEN"
        else:
            sidebar_style = SIDEBAR_STYLE
            content_style = CONTENT_STYLE
            cur_nclick = "SHOW"
    else:
        sidebar_style = SIDEBAR_STYLE
        content_style = CONTENT_STYLE
        cur_nclick = 'SHOW'

    return sidebar_style, content_style, cur_nclick


# this callback uses the current pathname to set the active state of the
# corresponding nav link to true, allowing users to tell see page they are on
@app.callback(
    [Output(f"page-{i}-link", "active") for i in range(1, 4)],
    [Input("url", "pathname")]
)
def toggle_active_links(pathname):
    if pathname == "/":
        # Treat page 1 as the homepage / index
        return True, False, False
    return [pathname == f"/page-{i}" for i in range(1, 4)]


@app.callback(Output('intermediate-value', 'data'), Input('dropdown-models', 'value'))
def make_calculations(value):
    if value is not None:
        incoming_batch_results, test_results, psi_table, psi_var_table_sum, psi_var_table_det, monitoring, scorecard, y_test, pd_X_test, stat_tests_report = helper_functions.calculate_data()
    else:
        # incoming_batch_results, test_results, psi_table, psi_var_table_sum, psi_var_table_det, y_test, pd_X_test = helper_functions.initialize_dash_vars()
        incoming_batch_results = {'binomial': 'N/A',
                                  'brier': 'N/A',
                                  'herfindahl': 'N/A',
                                  'hosmer': 'N/A',
                                  'spiegelhalter': 'N/A',
                                  'jeffreys': 'N/A',
                                  'roc_auc': 'N/A',
                                  'ber': 'N/A',
                                  'log_loss': 'N/A',
                                  'ks': 'N/A'}
        test_results = {'binomial': 'N/A',
                        'brier': 'N/A',
                        'herfindahl': 'N/A',
                        'hosmer': 'N/A',
                        'spiegelhalter': 'N/A',
                        'jeffreys': 'N/A',
                        'roc_auc': 'N/A',
                        'ber': 'N/A',
                        'log_loss': 'N/A',
                        'ks': 'N/A'}
        psi_table = None
        psi_var_table_sum = None
        psi_var_table_det = None
        y_test = None
        pd_X_test = None
        stat_tests_report = None

    incoming_batch_results = json.dumps(incoming_batch_results)
    test_results = json.dumps(test_results)

    if isinstance(psi_table, pd.DataFrame):
        psi_table = psi_table.to_json(date_format='iso', orient='split')
    else:
        psi_table = json.dumps(psi_table)

    if isinstance(psi_var_table_sum, pd.DataFrame):
        psi_var_table_sum = psi_var_table_sum.to_json(date_format='iso', orient='split')
    else:
        psi_var_table_sum = json.dumps(psi_var_table_sum)

    if isinstance(psi_var_table_det, pd.DataFrame):
        psi_var_table_det = psi_var_table_det.to_json(date_format='iso', orient='split')
    else:
        psi_var_table_det = json.dumps(psi_var_table_det)

    if y_test is None:
        y_test = json.dumps(y_test)
    else:
        y_test = json.dumps(y_test.tolist())

    if pd_X_test is None:
        pd_X_test = json.dumps(pd_X_test)
    else:
        pd_X_test = json.dumps(pd_X_test.tolist())

    stat_tests_report = json.dumps(stat_tests_report)

    datasets = {'incoming_batch_results': incoming_batch_results,
                'test_results': test_results,
                'psi_table': psi_table,
                'psi_var_table_sum': psi_var_table_sum,
                'psi_var_table_det': psi_var_table_det,
                'y_test': y_test,
                'pd_X_test': pd_X_test,
                'stat_tests_report': stat_tests_report}
    return json.dumps(datasets)


@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
    Input("intermediate-value", 'data')
)
def render_page_content(pathname, data):
    datasets = json.loads(data)

    incoming_batch_results = json.loads(datasets['incoming_batch_results'])
    test_results = json.loads(datasets['test_results'])

    if datasets['psi_table'] == 'null':
        psi_table = json.loads(datasets['psi_table'])
    else:
        psi_table = pd.read_json(datasets['psi_table'], orient='split')

    if datasets['psi_var_table_sum'] == 'null':
        psi_var_table_sum = json.loads(datasets['psi_var_table_sum'])
    else:
        psi_var_table_sum = pd.read_json(datasets['psi_var_table_sum'], orient='split')

    if datasets['psi_var_table_det'] == 'null':
        psi_var_table_det = json.loads(datasets['psi_var_table_det'])
    else:
        psi_var_table_det = pd.read_json(datasets['psi_var_table_det'], orient='split')

    y_test = json.loads(datasets['y_test'])
    pd_X_test = json.loads(datasets['pd_X_test'])
    stat_tests_report = json.loads(datasets['stat_tests_report'])

    if pathname in ["/", "/page-1"]:
        page_content = pages.accuracy.generate_accuracy_page(incoming_batch_results, test_results, y_test, pd_X_test)
        return page_content

    elif pathname == "/page-2":
        page_content = pages.stability.generate_stability_page(incoming_batch_results, test_results, psi_table,
                                                               psi_var_table_sum)

        return page_content
    elif pathname == "/page-3":
        page_content = html.H4(stat_tests_report)
        return page_content

    elif pathname == "/summary":
        page_content = html.Div([
            html.H2('Overall quality score: GOOD'),
            html.H3('AUC')
        ])
        return page_content

    elif pathname == '/report-generator':
        page_content = html.Div([

            html.H2('Generate report here')

        ])
        return page_content

    return dbc.Container(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ]
    )


if __name__ == "__main__":
    app.run_server(debug=True, port=8086)