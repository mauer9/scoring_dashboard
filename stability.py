from dash import dcc, html
import dash_bootstrap_components as dbc
from plotting_functions import new_create_card, psi_plot_ly, psi_variable_plot


def generate_stability_page(incoming_batch_results, test_results, psi_table, psi_var_table_sum):
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
                    new_create_card('Log Loss', '12414', incoming_batch_results['log_loss'],
                                    test_results['log_loss'])
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
                dcc.Graph(id='graph-with-slider', figure=psi_variable_plot(psi_var_table_sum))
            ),
            dbc.Col(
                dcc.Graph(id='graph-distributions',
                          figure=psi_plot_ly(psi_table)
                          )
            )
        ])
    ])
    return page_content
