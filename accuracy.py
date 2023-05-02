from dash import dcc, html
import dash_bootstrap_components as dbc
from plotting_functions import new_create_card, plot_auc_roc, plot_ks, no_figure_plot


def generate_accuracy_page(incoming_batch_results, test_results, y_test, pd_X_test):
    page_content = html.Div([
        dbc.Row(
            [
                dbc.Col(
                    new_create_card('Log Loss', '12412', incoming_batch_results['log_loss'],
                                    test_results['log_loss'])
                ),

                dbc.Col(
                    new_create_card('Brier', '12413', incoming_batch_results['brier'],
                                    test_results['brier'])
                ),

                dbc.Col(
                    new_create_card('K-S', '12414', incoming_batch_results['ks'],
                                    test_results['ks'])
                ),

                dbc.Col(
                    new_create_card('AUROC', '12415', incoming_batch_results['roc_auc'],
                                    test_results['roc_auc'])
                ),

                dbc.Col(
                    new_create_card('BER', '12416', incoming_batch_results['ber'],
                                    test_results['ber'])
                )
            ]),
        # dbc.Row([html.Div(dcc.Dropdown(X_test.columns, 'age', id='dropdown'),
        #                   style={'width': '50%'}
        #                   ),
        #          html.Div(dcc.Dropdown(X_test.columns, 'age', id='dropdown_2'),
        #                   style={'width': '50%'}
        #                   )
        #          ]
        #         ),
        dbc.Row([

            dbc.Col(
                dcc.Graph(id='logloss-plot',
                          figure=no_figure_plot()
                          ),
                width=5
            ),
            dbc.Col(
                dcc.Graph(id='graph-with-slider',
                          figure=plot_auc_roc(y_test, pd_X_test)
                          ),
                width=3
            ),
            dbc.Col(
                dcc.Graph(id='graph-distributions',
                          figure=plot_ks(y_test, pd_X_test)
                          ),
                width=3
            )
        ])
    ])
    return page_content