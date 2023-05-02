from optbinning.binning import binning_process
from optbinning.scorecard.scorecard import _check_parameters
from optbinning.scorecard.scorecard import _compute_scorecard_points
from sklearn.base import TransformerMixin
import joblib
import pandas as pd
import numpy as np
import credit_py_validation as cpv

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from optbinning.scorecard import Scorecard
from optbinning.logging import Logger

import time
from sklearn.base import clone
from sklearn.utils.multiclass import type_of_target
import gitlab
import sqlite3
import json
import io
from contextlib import redirect_stdout

from optbinning import BinningProcess
from optbinning.scorecard import ScorecardMonitoring

logger = Logger(__name__).logger


def capture_output(output_stream):
    f = io.StringIO()
    with redirect_stdout(f):
        output_stream
        # output_stream.system_stability_report()
    output = f.getvalue()
    return output


def get_all_models():
    gl = gitlab.Gitlab('http://gitlab.cloud.halykbank.nb', private_token='yBKJTSeuciukPxTfWJ-Z',
                       ssl_verify=False, api_version=4)
    project = gl.projects.get(2370)
    repo = project.repository_tree(ref='master')
    directories = [item["path"] for item in repo if item["type"] == "tree"]
    return directories


def conduct_tests(table_for_tests):  # __REWORK - add options to choose model and different tests
    # if self.table_for_tests is None:
    #     raise Exception('Dataset for tests is not ready')
    binomial_result = cpv.binomial_test(table_for_tests, 'Credit Rating', 'Default Flag',
                                        'Default Probability').to_dict()
    brier_score_result = cpv.brier_score(table_for_tests, 'Credit Rating', 'Default Flag',
                                         'Default Probability')
    herfindahl_result = cpv.herfindahl_test(table_for_tests, 'Credit Rating')
    hosmer_result = cpv.hosmer_test(table_for_tests, 'Credit Rating', 'Default Flag', 'Default Probability')
    spiegelhalter_result = cpv.spiegelhalter_test(table_for_tests, 'Credit Rating', 'Default Flag',
                                                  'Default Probability')
    jeffreys_result = cpv.jeffreys_test(table_for_tests, 'Credit Rating', 'Default Flag',
                                        'Default Probability').to_dict()
    roc_auc_result = cpv.roc_auc(table_for_tests, 'Default Flag', 'Default Probability')  # FIX - outputs 1.0 AUC
    ber_result = cpv.bayesian_error_rate(table_for_tests['Default Flag'],
                                         table_for_tests['Default Probability'])
    log_loss_result = log_loss(table_for_tests['Default Flag'], table_for_tests['Default Probability'])

    ks_result = cpv.kolmogorov_smirnov(table_for_tests['Default Flag'].reset_index(drop=True),
                                       table_for_tests['Default Probability'].reset_index(drop=True))
    # cpv.psi(loans_test, 'Credit Rating', 'PD')

    test_results = {'binomial': binomial_result,
                    'brier': brier_score_result,
                    'herfindahl': herfindahl_result,
                    'hosmer': hosmer_result,
                    'spiegelhalter': spiegelhalter_result,
                    'jeffreys': jeffreys_result,
                    'roc_auc': roc_auc_result,
                    'ber': ber_result,
                    'log_loss': log_loss_result,
                    'ks': ks_result}
    return test_results


def initialize_dash_vars():
    init_cols = ['Bin', 'Count A', 'Count E', 'Count A (%)', 'Count E (%)', 'PSI']
    stat_test_results = {'binomial': 'N/A',
                         'brier': 'N/A',
                         'herfindahl': 'N/A',
                         'hosmer': 'N/A',
                         'spiegelhalter': 'N/A',
                         'jeffreys': 'N/A',
                         'roc_auc': 'N/A',
                         'ber': 'N/A',
                         'log_loss': 'N/A'}
    psi_table = pd.DataFrame(columns=init_cols)
    psi_var_table_sum = pd.DataFrame(columns=init_cols)
    psi_var_table_det = pd.DataFrame(columns=init_cols)
    y_test = pd.Series()
    pd_X_test = pd.Series()
    return stat_test_results, stat_test_results, psi_table, psi_var_table_sum, psi_var_table_det, y_test, pd_X_test


def load_full_simulation_df():  # TEMPORARY FUNCTION FOR SIMULATION
    con = sqlite3.connect('new_credit_data.db')
    query_X = """SELECT *
    FROM X_train_log
    """
    query_y = """SELECT *
    FROM y_train_log
    """
    X = pd.read_sql(query_X, con, index_col=None)
    y = pd.read_sql(query_y, con, index_col=None)
    y = y.values.ravel()
    X_train, X_unseen, y_train, y_unseen = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_unseen, y_train, y_unseen


def load_model():
    with open('binning_fit_params.json') as json_file:
        binning_fit_params = json.load(json_file)
    # Loading the model
    lr_model = joblib.load('lr_model.pkl')
    return lr_model, binning_fit_params


def calculate_data():  # RE -FUCKING- DO IT
    X_train, X_unseen, y_train, y_unseen = load_full_simulation_df()
    lr_model, binning_fit_params = load_model()

    variable_names = list(X_train.columns)
    binning_process = BinningProcess(variable_names,  # selection_criteria=selection_criteria,
                                     binning_fit_params=binning_fit_params)
    scorecard = AutoPrepareScoreCard(binning_process=binning_process,
                                     estimator=lr_model, scaling_method="min_max",
                                     scaling_method_params={"min": 300, "max": 850}, verbose=True)

    scorecard.fit(X_train, y_train)
    monitoring = ScorecardMonitoring(scorecard=scorecard, psi_method="cart",
                                     psi_n_bins=10, verbose=True)

    ff = io.StringIO()
    with redirect_stdout(ff):
        monitoring.fit(X_unseen, y_unseen, X_train, y_train)
    asd2 = ff.getvalue()

    stat_tests_report = capture_output(monitoring.system_stability_report())

    # Calculate train
    pd_X_train = scorecard.predict_proba(X_train)[:, 1]
    default_flag_X_train = scorecard.predict(X_train)
    score_X_train = scorecard.score(X_train)
    rating_X_train = scorecard.get_credit_ratings(score_X_train)

    X_train['Default Probability'] = pd_X_train
    X_train['Credit Rating'] = rating_X_train
    X_train['Default Flag'] = default_flag_X_train

    # Calculate test
    pd_X_test = scorecard.predict_proba(X_unseen)[:, 1]
    default_flag_X_test = scorecard.predict(X_unseen)
    score_X_test = scorecard.score(X_unseen)
    rating_X_test = scorecard.get_credit_ratings(score_X_test)

    print('pd_X_Test: ', pd_X_test)
    print('def_X_test: ', default_flag_X_test)
    unique, counts = np.unique(default_flag_X_test, return_counts=True)
    print(np.asarray((unique, counts)).T)

    X_unseen['Default Probability'] = pd_X_test
    X_unseen['Credit Rating'] = rating_X_test
    X_unseen['Default Flag'] = default_flag_X_test

    incoming_batch_results = conduct_tests(X_unseen)
    test_results = conduct_tests(X_train)
    psi_var_table_det = monitoring.psi_variable_table(style='detailed')
    psi_var_table_sum = monitoring.psi_variable_table(style='summary')
    psi_table = monitoring.psi_table()

    return incoming_batch_results, test_results, psi_table, psi_var_table_sum, psi_var_table_det, monitoring, scorecard, y_unseen, pd_X_test, stat_tests_report


# CHANGE ALL
class AutoPrepareScoreCard(Scorecard):
    def __init__(self, binning_process, estimator, scaling_method=None, scaling_method_params=None,
                 intercept_based=False, reverse_scorecard=False, rounding=False, verbose=False, db_connection=None,
                 target_name='target'):  # ADD PARAMETERS
        super().__init__(binning_process, estimator, scaling_method=None, scaling_method_params=None,
                         intercept_based=False, reverse_scorecard=False, rounding=False, verbose=False)

        self.binning_process = binning_process
        self.estimator = estimator
        self.scaling_method = scaling_method
        self.scaling_method_params = scaling_method_params
        self.intercept_based = intercept_based
        self.reverse_scorecard = reverse_scorecard
        self.rounding = rounding
        self.verbose = verbose

        self.incoming_labels = None
        self.test_data_tests = None
        self.incoming_data_tests = None
        self.incoming_data = None
        self.db_connection = db_connection
        self.target_name = target_name

        self.train_set = None
        self.test_set = None
        self.pipeline = None

        self.test_predictions = None

        self.choices = list(reversed(range(1, 11)))
        self.loan_type = 'УЗП'

        self.credit_scores = None
        self.credit_ratings = None

        # attributes
        self.binning_process_ = None
        self.estimator_ = None
        self.intercept_ = 0

        self._metric_special = None
        self._metric_missing = None

        # auxiliary
        self._target_dtype = None

        # timing
        self._time_total = None
        self._time_binning_process = None
        self._time_estimator = None
        self._time_build_scorecard = None
        self._time_rounding = None

        self._is_fitted = False

    def fit(self, X, y, sample_weight=None, metric_special=0, metric_missing=0,
            show_digits=2, check_input=False, fitted=True):
        """Fit scorecard.
        Parameters
        ----------
        X : pandas.DataFrame (n_samples, n_features)
            Training vector, where n_samples is the number of samples.
        y : array-like of shape (n_samples,)
            Target vector relative to x.
        sample_weight : array-like of shape (n_samples,) (default=None)
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.
            This option is only available for a binary target.
        metric_special : float or str (default=0)
            The metric value to transform special codes in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate, and any numerical value.
        metric_missing : float or str (default=0)
            The metric value to transform missing values in the input vector.
            Supported metrics are "empirical" to use the empirical WoE or
            event rate and any numerical value.
        check_input : bool (default=False)
            Whether to check input arrays.
        show_digits : int, optional (default=2)
            The number of significant digits of the bin column.
        fitted : boolean, optional (default=True)
            If the estimator is fitted or not
        Returns
        -------
        self : Scorecard
            Fitted scorecard.
        """
        return self._fit(X, y, sample_weight, metric_special, metric_missing,
                         show_digits, check_input, fitted)

    def _fit(self, X, y, sample_weight, metric_special, metric_missing,
             show_digits, check_input, fitted=True):

        # Store the metrics for missing and special bins for predictions
        self._metric_special = metric_special
        self._metric_missing = metric_missing

        time_init = time.perf_counter()

        # if self.verbose:
        #     logger.info("Scorecard building process started.")
        #     logger.info("Options: check parameters.")

        # _check_parameters(**self.get_params(deep=False))

        # Check X dtype
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas.DataFrame.")

        # Target type and metric
        self._target_dtype = type_of_target(y)

        if self._target_dtype not in ("binary", "continuous"):
            raise ValueError("Target type {} is not supported."
                             .format(self._target_dtype))

        # _check_scorecard_scaling(self.scaling_method,
        #                          self.scaling_method_params,
        #                          self.rounding,
        #                          self._target_dtype)

        # # Check sample weight
        # if sample_weight is not None and self._target_dtype != "binary":
        #     raise ValueError("Target type {} does not support sample weight."
        #                      .format(self._target_dtype))
        metric = 'woe'
        bt_metric = 'WoE'
        if self._target_dtype == "binary":
            metric = "woe"
            bt_metric = "WoE"
        elif self._target_dtype == "continuous":
            metric = "mean"
            bt_metric = "Mean"
        # if self.verbose:
        #     logger.info("Dataset: {} target.".format(self._target_dtype))

        # Fit binning process
        if self.verbose:
            logger.info("Binning process started.")

        time_binning_process = time.perf_counter()
        self.binning_process_ = clone(self.binning_process)

        # Suppress binning process verbosity
        self.binning_process_.set_params(verbose=False)

        X_t = self.binning_process_.fit_transform(
            X[self.binning_process.variable_names], y, sample_weight, metric,
            metric_special, metric_missing, show_digits, check_input)

        self._time_binning_process = time.perf_counter() - time_binning_process

        # if self.verbose:
        #     logger.info("Binning process terminated. Time: {:.4f}s"
        #                 .format(self._time_binning_process))

        if not fitted:
            # Fit estimator
            time_estimator = time.perf_counter()
            if self.verbose:
                logger.info("Fitting estimator.")

            self.estimator_ = clone(self.estimator)
            self.estimator_.fit(X_t, y, sample_weight)

            self._time_estimator = time.perf_counter() - time_estimator

            if self.verbose:
                logger.info("Fitting terminated. Time {:.4f}s"
                            .format(self._time_estimator))
        else:
            from copy import deepcopy
            time_estimator = time.perf_counter()
            self.estimator_ = deepcopy(self.estimator)
            self._time_estimator = time.perf_counter() - time_estimator

        # Get coefs
        intercept = 0
        if hasattr(self.estimator_, 'coef_'):
            coefs = self.estimator_.coef_.flatten()
            if hasattr(self.estimator_, 'intercept_'):
                intercept = self.estimator_.intercept_
        else:
            raise RuntimeError('The classifier does not expose '
                               '"coef_" attribute.')

        # Build scorecard
        time_build_scorecard = time.perf_counter()

        if self.verbose:
            logger.info("Scorecard table building started.")

        selected_variables = self.binning_process_.get_support(names=True)
        binning_tables = []
        for i, variable in enumerate(selected_variables):
            optb = self.binning_process_.get_binned_variable(variable)
            binning_table = optb.binning_table.build(
                show_digits=show_digits, add_totals=False)

            c = coefs[i]
            binning_table.loc[:, "Variable"] = variable
            binning_table.loc[:, "Coefficient"] = c
            binning_table.loc[:, "Points"] = binning_table[bt_metric] * c

            nt = len(binning_table)
            if metric_special != 'empirical':
                if isinstance(optb.special_codes, dict):
                    n_specials = len(optb.special_codes)
                else:
                    n_specials = 1

                binning_table.loc[
                nt - 1 - n_specials:nt - 2, "Points"] = metric_special * c
            elif metric_missing != 'empirical':
                binning_table.loc[nt - 1, "Points"] = metric_missing * c

            binning_table.index.names = ['Bin id']
            binning_table.reset_index(level=0, inplace=True)
            binning_tables.append(binning_table)

        df_scorecard = pd.concat(binning_tables)
        df_scorecard.reset_index()

        # Apply score points
        if self.scaling_method is not None:
            points = df_scorecard["Points"]
            scaled_points = _compute_scorecard_points(
                points, binning_tables, self.scaling_method,
                self.scaling_method_params, intercept, self.reverse_scorecard)

            df_scorecard.loc[:, "Points"] = scaled_points

        if self.intercept_based:
            scaled_points, self.intercept_ = _compute_intercept_based(
                df_scorecard)
            df_scorecard.loc[:, "Points"] = scaled_points

        time_rounding = time.perf_counter()
        if self.rounding:
            points = df_scorecard["Points"]
            if self.scaling_method in ("pdo_odds", None):
                round_points = np.rint(points)

                if self.intercept_based:
                    self.intercept_ = np.rint(self.intercept_)
            elif self.scaling_method == "min_max":
                round_mip = RoundingMIP()
                round_mip.build_model(df_scorecard)
                status, round_points = round_mip.solve()

                if status not in ("OPTIMAL", "FEASIBLE"):
                    if self.verbose:
                        logger.warning("MIP rounding failed, method nearest "
                                       "integer used instead.")
                    # Back-up method
                    round_points = np.rint(points)

                if self.intercept_based:
                    self.intercept_ = np.rint(self.intercept_)

            df_scorecard.loc[:, "Points"] = round_points
        self._time_rounding = time.perf_counter() - time_rounding

        self._df_scorecard = df_scorecard

        self._time_build_scorecard = time.perf_counter() - time_build_scorecard
        self._time_total = time.perf_counter() - time_init

        if self.verbose:
            logger.info("Scorecard table terminated. Time: {:.4f}s"
                        .format(self._time_build_scorecard))
            logger.info("Scorecard building process terminated. Time: {:.4f}s"
                        .format(self._time_total))

        # Completed successfully
        self._is_fitted = True

        return self

    def read_pipeline(self, path, from_file=True, from_db=False):  # __REDO - for different model formats
        if from_file:
            self.pipeline = joblib.load(path)
            return self.pipeline
        return None

    def _read_data_db(self, query):
        return pd.read_sql(query, self.db_connection, index_col=None)

    def read_data(self, train_query, test_query):
        self.train_set = self._read_data_db(train_query)
        self.test_set = self._read_data_db(test_query)
        return self.train_set, self.test_set

    def read_incoming_data(self, incoming_query, incoming_labels_query=None):
        self.incoming_data = self._read_data_db(incoming_query)
        if incoming_labels_query is not None:
            self.incoming_labels = self._read_data_db(incoming_labels_query)
            self.incoming_data[self.target_name] = self._read_data_db(incoming_labels_query)[self.target_name].copy()
        return self.incoming_data

    def make_predictions(self, df_set, proba=True):
        if self.pipeline is None:
            raise Exception('No pipeline is set to predict')
        if proba:
            return self.pipeline.predict_proba(df_set.drop(self.target_name, axis=1, errors='ignore'))[:, 1]
        return self.pipeline.predict(df_set.drop(self.target_name, axis=1, errors='ignore'))

    def set_score_card(self, choices, loan_type):  # __REWORK - add actual scorecard
        self.choices = choices
        self.loan_type = loan_type

    def get_credit_scores(self, df):  # __REWORK - add scoring card construction options (can be non-linear)
        if df is None:
            raise Exception('Data set is not specified')
        default_probs = self.make_predictions(df, proba=True)
        minmax_sc = MinMaxScaler(feature_range=(300, 850))
        minmax_sc.fit(default_probs.reshape(-1, 1))
        credit_scores = minmax_sc.transform(default_probs.reshape(-1, 1)).ravel().round()
        self.credit_scores = credit_scores
        return credit_scores

    def get_credit_ratings(self, credit_scores):
        if self.choices is None:
            raise Exception('Need to set score card first')
        if self.loan_type is None:
            raise Exception('Please specify loan type first')
        choices = self.choices.copy()
        if not isinstance(credit_scores, pd.Series):
            credit_scores = pd.Series(credit_scores)
        if self.loan_type == 'УЗП' or self.loan_type == 'НЗП':
            cond_list = [
                credit_scores.lt(576),
                credit_scores.between(576, 585),
                credit_scores.between(586, 594),
                credit_scores.between(595, 603),
                credit_scores.between(604, 610),
                credit_scores.between(611, 617),
                credit_scores.between(618, 625),
                credit_scores.between(626, 633),
                credit_scores.between(634, 642),
                credit_scores.gt(642)]
        #     else if loan_type == 'Пенсионная':
        #         cond_list = [
        #             df_series.lt(1),
        #             df_series.between(1, 30),
        #             df_series.between(31, 60),
        #             df_series.between(61, 90),
        #             df_series.between(91, 180),
        #             df_series.gt(180)]
        credit_ratings = np.select(cond_list, choices)

        self.credit_ratings = credit_ratings
        return credit_ratings

    def prepare_test_tests(self):
        if self.test_set is None:
            raise Exception('Test set is missing')
        else:
            test_data_tests = self.test_set.copy()
        test_data_tests['Credit Rating'] = self.credit_ratings.copy()
        test_data_tests.rename(columns={self.target_name: 'Default Flag'}, inplace=True)
        test_data_tests['Default Probability'] = self.make_predictions(self.test_set, proba=True)
        self.test_data_tests = test_data_tests.copy()
        return test_data_tests

    def prepare_incoming_tests(self):
        if self.incoming_data is None:
            raise Exception('Incoming data is missing')
        else:
            incoming_data_tests = self.incoming_data.copy()
        incoming_data_tests['Credit Rating'] = self.credit_ratings.copy()
        incoming_data_tests.rename(columns={'target': 'Default Flag'}, inplace=True)
        incoming_data_tests['Default Probability'] = self.make_predictions(self.incoming_data, proba=True)
        self.incoming_data_tests = incoming_data_tests.copy()
        return incoming_data_tests


# --------------------------------------------------------------------------CUSTOM TRANSFORMERS FOR PICKLE FILE

def col_dropper(df):
    df = df.drop(['INCOME_CALC_METHOD', 'SUM_PROCENTY_TODAY_Sum'], axis=1)
    return df


def fix_dtypes(df):
    df['USTUPKA_DEISTV'] = df['USTUPKA_DEISTV'].astype(int)
    df['USTUPKA_ZAKRYTYE'] = df['USTUPKA_ZAKRYTYE'].astype(int)
    df['GENDER'] = df['GENDER'].astype(int)
    df['CREDIT_HISTORY'] = df['CREDIT_HISTORY'].astype(int)
    df['SUBJ_STATUS'] = df['SUBJ_STATUS'].astype(int)
    df['age'] = df['age'].astype(int)
    return df


class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value
        in column.

        Columns of other types are imputed with mean of column.

        """

    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
                               if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
                              index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
# --------------------------------------------------------------------------CUSTOM TRANSFORMERS FOR PICKLE FILE