import sys

from ds_utils.db.connectors import PivotDW, HealthcareDW
from ds_utils.db.generic_sql import drop_table

schema = 'data_science'

blocks_file_s3 = 's3://hc-interim-data/GeoIP2-City-Blocks-IPv4-opt.csv'
blocks_table = 'maxmind_ipv4_geo_blocks'

locations_file_s3 = 's3://hc-interim-data/GeoIP2-City-Locations-en.csv'
locations_table = 'maxmind_geo_locations'

local_blocks_path = '/Users/jdelvalle/Downloads/GeoIP2-City-CSV_20210702/GeoIP2-City-Blocks-IPv4-opt.csv'
local_locations_path = '/Users/jdelvalle/Downloads/GeoIP2-City-CSV_20210702/GeoIP2-City-Locations-en.csv'

blocks_drop = drop_table(schema=schema, table=blocks_table)
locations_drop = drop_table(schema=schema, table=locations_table)

blocks_create = f"""
    CREATE TABLE IF NOT EXISTS {schema}.{blocks_table} (
        netowrk_index VARCHAR,
        start_int BIGINT,
        end_int BIGINT,
        network VARCHAR,
        maxmind_id INTEGER,
        registered_country_id INTEGER,
        represented_country_id INTEGER,
        is_anonymous_proxy BOOLEAN,
        is_satellite_provider BOOLEAN,
        postal_code VARCHAR,
        latitude NUMERIC,
        longitude NUMERIC,
        accuracy_radius INTEGER
        )
    ;
"""

locations_create = f"""
    CREATE TABLE IF NOT EXISTS {schema}.{locations_table} (
        maxmind_id INTEGER,
        locale_code VARCHAR,
        continent_code VARCHAR,
        continent_name VARCHAR,
        country_iso_code VARCHAR,
        country_name VARCHAR,
        subdivision_1_iso_code VARCHAR,
        subdivision_1_name VARCHAR,
        subdivision_2_iso_code VARCHAR,
        subdivision_2_name VARCHAR,
        city_name VARCHAR,
        metro_code VARCHAR,
        time_zone VARCHAR,
        is_in_european_union BOOLEAN
        )
    DISTSTYLE ALL
    SORTKEY (maxmind_id)
    ;
"""

if __name__ == '__main__':

    if sys.argv[-1] == 'PH':
        connector = PivotDW
    elif sys.argv[-1] == 'HC':
        connector = HealthcareDW
    else:
        print("Bad argument. Example usage:\n\tpython redshift_load.py PH\n\tpython redshfit_load.py HC")
        sys.exit()

    sql_commands = [
        blocks_drop,
        locations_drop,
        blocks_create,
        locations_create,
    ]

    with connector() as db_context:
        for command in sql_commands:
            db_context.exec(command)

        # Set `drop_file` to `False` if files is too large you're afraid loading might fail
        db_context.load_csv(local_blocks_path, schema, blocks_table, skip_header=True, drop_file=False)
        db_context.load_csv(local_locations_path, schema, locations_table, skip_header=True, drop_file=False)

