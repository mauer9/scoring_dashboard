# # # Database
# # import oracledb
# # import sqlite3
# # import sqlalchemy
# # # Data manipulation
# # import pandas as pd
# # from helper_functions import col_dropper, fix_dtypes, DataFrameImputer, AutoPrepareScoreCard
# # import numpy as np
# # from optbinning import BinningProcess
# # from optbinning import Scorecard
# # from optbinning.scorecard import ScorecardMonitoring
# #
# # import joblib
# #
# # con = sqlite3.connect('credit_data.db')
# # cur = con.cursor()
# #
# #
# # def read_pipeline(path):  # __REDO - for different model formats
# #     pipeline = joblib.load(path)
# #     return pipeline
# #
# #
# # estimator = read_pipeline('clf_pipeline.pkl')
# #
# #
# #
# # def list_all_table():
# #     query = """SELECT name FROM sqlite_master WHERE type='table';"""
# #     print(pd.read_sql(query, con, index_col=None))
# #     return None
# #
# #
# # query_inc = """SELECT *
# # FROM test_set
# # """
# # test_set = pd.read_sql(query_inc, con, index_col=None)
# #
# # scorecard = Scorecard()
#
#
# import datetime
# import dash_bootstrap_components as dbc
# from dash import html, Dash
#
# app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
#
# # app.layout = html.H1('The time is: ' + str(datetime.datetime.now()))
# #
# # if __name__ == '__main__':
# #     app.run_server(debug=True)
#
# def serve_layout():
#     return html.H1('The time is: ' + str(datetime.datetime.now()))
#
# app.layout = serve_layout
#
# if __name__ == '__main__':
#     app.run_server(debug=True)