# 

from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px

from plotting_functions import plot_hist_dist, plot_feature_importances, create_card, psi_plot_ly, plot_auc_roc, plot_ks, psi_variable_plot
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

import credit_py_validation as cpv

from plotly.tools import mpl_to_plotly

# Global variables
LOGO_PATH = 'assets/halyk_logo.png'

# Database connection
con = sqlite3.connect('new_credit_data.db')
cur = con.cursor()
query_X = """SELECT *
FROM X_train_log
"""
X_train_log = pd.read_sql(query_X, con, index_col=None)

query_y = """SELECT *
FROM y_train_log
"""
y_train_log = pd.read_sql(query_y, con, index_col=None)
X_train, X_test, y_train, y_test = train_test_split(X_train_log, y_train_log, test_size=0.2, random_state=42)
def do_calculations():
    # Read binning fit params
    with open('binning_fit_params.json') as json_file:
        binning_fit_params = json.load(json_file)

    # Loading the model
    lr_model = joblib.load('lr_model.pkl')

    # selection_criteria = {
    #     "iv": {"min": 0.02, "max": 1},
    #     "quality_score": {"min": 0.01}
    # }
    variable_names = list(X_train.columns)

    binning_process = BinningProcess(variable_names,  # selection_criteria=selection_criteria,
                                     binning_fit_params=binning_fit_params)

    scorecard = helper_functions.AutoPrepareScoreCard(binning_process=binning_process,
                                                      estimator=lr_model, scaling_method="min_max",
                                                      scaling_method_params={"min": 300, "max": 850}, verbose=True)

    scorecard.fit(X_train, y_train.values.ravel())

    monitoring = ScorecardMonitoring(scorecard=scorecard, psi_method="cart",
                                     psi_n_bins=10, verbose=True)

    monitoring.fit(X_test, y_test.values.ravel(), X_train, y_train.values.ravel())

    monitoring.psi_table()

    # Calculate train
    pd_X_train = scorecard.predict_proba(X_train)[:, 1]
    default_flag_X_train = scorecard.predict(X_train)
    score_X_train = scorecard.score(X_train)
    rating_X_train = scorecard.get_credit_ratings(score_X_train)

    X_train['Default Probability'] = pd_X_train
    X_train['Credit Rating'] = rating_X_train
    X_train['Default Flag'] = default_flag_X_train

    # Calculate test
    pd_X_test = scorecard.predict_proba(X_test)[:, 1]
    default_flag_X_test = scorecard.predict(X_test)
    score_X_test = scorecard.score(X_test)
    rating_X_test = scorecard.get_credit_ratings(score_X_test)

    X_test['Default Probability'] = pd_X_test
    X_test['Credit Rating'] = rating_X_test
    X_test['Default Flag'] = default_flag_X_test

    incoming_batch_results = helper_functions.conduct_tests(X_test)
    test_results = helper_functions.conduct_tests(X_train)
    # card_tests = ['log_loss', 'brier', 'roc_auc', 'ber']
    return incoming_batch_results, test_results, scorecard, monitoring

# # plot_auc_roc(y_train, scorecard.predict_proba(X_train)[:, 0])))
# def get_change(current, previous):
#     if current == previous:
#         return 0
#     try:
#         return (abs(current - previous) / previous) * 100.0
#     except ZeroDivisionError:
#         return float('inf')
# tests_percent_change = {k + '_change': get_change(test_results[k], incoming_batch_results[k]) for k in card_tests}



app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

# app.layout = html.Div([
#     html.H2('Welcome to HB Credit Risk Model Monitoring System!'),
#     html.H3('Please, choose a model from the dropdown menu below: '),
#     # html.Div(dcc.dropdown)
#     html.Button(id='calculate-button-state', n_clicks=0, children='Submit')
# ])

###### RE-WORK - initilize arrays and values for empty DASHBOARD!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

incoming_batch_results, test_results, scorecard, monitoring = do_calculations() # CHANGE TO PRODUCE PLACEHOLDERS

# @app.callback(
#     Output('graph-distributions', 'figure'),
#     Input('dropdown', 'value')
# )
# def update_layout(value):
#     fig = plot_hist_dist(train[value], test[value])
#     return fig


app.layout = html.Div([
    # date_range(),
    html.Br(),
    html.Div(id='output-container-date-picker-range'),
    dbc.Row([html.Img(src=LOGO_PATH, style={'height': '10%', 'width': '10%'})]),
    html.H2('Welcome to HB Credit Risk Model Monitoring System!'),
    html.Br(),

    dcc.Tabs([
        dcc.Tab(label='Assess accuracy', children=[
            dbc.Row(
                [dbc.Col(create_card(incoming_batch_results['log_loss'], '1', 'Log Loss',
                                     f'Initial: {test_results["log_loss"]:.3f}')),
                 dbc.Col(
                     create_card(incoming_batch_results['brier'], '2', 'Brier',
                                 f'Initial: {test_results["brier"]:.3f}')),
                 dbc.Col(
                     create_card(incoming_batch_results['roc_auc'], '3', 'AUROC',
                                 f'Initial: {test_results["roc_auc"]:.3f}')),
                 dbc.Col(create_card(incoming_batch_results['ber'], '4', 'BER', f'Initial: {test_results["ber"]:.3f}')),
                 dbc.Col(create_card(11, '5', 'K-S', 12))
                 ]),

            dbc.Row([html.Div(dcc.Dropdown(X_test.columns, 'age', id='dropdown'),
                              style={'width': '50%'}
                              ),
                     html.Div(dcc.Dropdown(X_test.columns, 'age', id='dropdown_2'),
                              style={'width': '50%'}
                              )
                     ]
                    ),
            dbc.Row([
                dbc.Col(
                    dcc.Graph(id='graph-with-slider', figure=psi_variable_plot(monitoring)
                              )
                ),
                dbc.Col(
                    dcc.Graph(id='graph-distributions',
                              figure=plot_ks(y_train.values.ravel(), scorecard.predict_proba(X_train)[:, 0])
                              )
                )
            ])
        ]),

        dcc.Tab(label='Assess stability', children=[
            dbc.Row(html.Br())
        ])
    ]),


    html.Br(),
    html.Br(),
    html.Div(html.Button('Generate report', id='generate-report-button', n_clicks=0))
])

if __name__ == '__main__':
    app.run_server(debug=True)


# @app.callback(
#     Output('graph-distributions', 'figure'),
#     Input('dropdown', 'value')
# )
# def update_hist_dist(value):
#     fig = plot_hist_dist(train[value], test[value])
#     return fig


# fig_roc_auc = plot_auc_roc(y_train, scorecard.predict_proba(X_train)[:, 0])