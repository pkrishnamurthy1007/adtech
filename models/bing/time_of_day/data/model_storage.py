import sys
from psycopg2.extras import RealDictCursor
import ast

sys.path.append('/Users/trevor/git/datascience-utils')
from db.redshift import RedshiftContextHc

def store_parameters(model_run_epoch, start_date, end_date, hour_resolution, segments_per_day,
    rps_spline_k, rps_spline_s, sessions_spline_k, sessions_spline_s
    ):

    create = """
        CREATE TABLE IF NOT EXISTS data_science.time_of_day_model_runs (
            model_run_epoch BIGINT,
            start_date DATE,
            end_date DATE,
            hour_resolution NUMERIC(5,4),
            segments_per_day INT,
            rps_spline_k INT,
            rps_spline_s INT,
            sessions_spline_k INT,
            sessions_spline_s INT,
            optimized_intervals VARCHAR(1000)
            )
        ;
    """

    insert = f"""
        INSERT INTO data_science.time_of_day_model_runs VALUES (
            {model_run_epoch},
            '{start_date}',
            '{end_date}',
            {hour_resolution},
            {segments_per_day},
            {rps_spline_k},
            {rps_spline_s},
            {sessions_spline_k},
            {sessions_spline_s}
            )
        ;
    """
    with RedshiftContextHc() as db_context:
        db_context.exec(create)
        db_context.exec(insert)

def store_optimized_set(model_run_epoch, optimized_intervals):

    update = f"""
        UPDATE data_science.time_of_day_model_runs
        SET optimized_intervals = '{optimized_intervals}'
        WHERE model_run_epoch = {model_run_epoch}
        ;
    """
    with RedshiftContextHc() as db_context:
        db_context.exec(update)


def load_parameters(model_run_epoch):

    select = f"""
        SELECT
            model_run_epoch,
            start_date,
            end_date,
            hour_resolution,
            segments_per_day,
            rps_spline_k,
            rps_spline_s,
            sessions_spline_k,
            sessions_spline_s
        FROM data_science.time_of_day_model_runs
        WHERE model_run_epoch = {model_run_epoch}
        ;
    """
    with RedshiftContextHc() as db_context:
        config = db_context.fetch(select, cursor_factory=RealDictCursor)

    return config[0]

def get_optimized_set(model_run_epoch):

    select = f"""
        SELECT optimized_intervals
        FROM data_science.time_of_day_model_runs
        WHERE model_run_epoch = {model_run_epoch}
        ;
    """
    with RedshiftContextHc() as db_context:
        optimized_set = db_context.fetch(select)

    return ast.literal_eval(optimized_set[0][0])