from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc


def generate_search_bar():
    search_bar = dbc.Row(
        [
            dbc.Col(dbc.Input(type="search", placeholder="Search")),
            dbc.Col(
                dbc.Button(
                    "Search", color="primary", className="ms-2", n_clicks=0
                ),
                width="auto",
            ),
        ],
        className="g-0 ms-auto flex-nowrap mt-3 mt-md-0",
        align="center",
    )
    return search_bar


def generate_navbar(model_list, logo):
    navbar = dbc.Navbar(
        dbc.Container(
            [
                html.A(
                    # Use row and col to control vertical alignment of logo / brand
                    dbc.Row(
                        [
                            dbc.Col(html.Img(src=logo, height='30px')),
                            dbc.Col(dbc.NavbarBrand(
                                "Credit Risk Model Monitoring System", className="ms-2")),
                        ],
                        align="left",
                        className="g-0",
                    ),
                    # href="https://halykbank.kz/",
                    style={"textDecoration": "none"},
                ),
                dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
                html.Div(dcc.Dropdown(model_list, id='dropdown-models', placeholder='Select a model'),
                         style={'width': '20%'}),
                dbc.Button("Sidebar", outline=True, color="secondary",
                           className="mr-1", id="btn_sidebar"),
                dbc.Collapse(
                    generate_search_bar(),
                    id="navbar-collapse",
                    is_open=False,
                    navbar=True,
                ),
            ]
        ),
        color = "dark",
        dark = True,
    )
    return navbar


def generate_sidebar(SIDEBAR_STYLE):
    sidebar=html.Div(
        [
            html.H3("CREMOSYS", className="display-4"),
            html.Hr(),
            html.P(
                "CREdit Risk Model MOnitoring SYStem is for a live observation of model's quality", className="lead"
            ),
            dbc.Nav(
                [
                    dbc.NavLink("Summary", href="/summary", id="summary-link"),
                    dbc.NavLink("Assess accuracy",
                                href="/page-1", id="page-1-link"),
                    dbc.NavLink("Assess stability",
                                href="/page-2", id="page-2-link"),
                    dbc.NavLink("Statistical tests report",
                                href="/page-3", id="page-3-link"),
                    dbc.NavLink(
                        "Generate report", href="/report-generator", id="report-generator-link"),
                ],
                vertical=True,
                pills=True,
            ),
        ],
        id = "sidebar",
        style = SIDEBAR_STYLE,
    )
    return sidebar


def generate_footer(FOOTER_STYLE):
    footer=html.Footer(
        id = 'footer',
        children = [
            html.H6(
                "Copyright " + u"\u00A9" + " 2023 Halyk Bank. All rights reserved. Developed by Janysbek Kusmangaliyev")
        ],
        style = FOOTER_STYLE
    )
    return footer
