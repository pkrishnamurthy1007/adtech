from db.redshift import RedshiftContextHc
from sql import DROP


schema = 'data_science'

blocks_file_s3 = 's3://hc-interim-data/GeoIP2-City-Blocks-IPv4-opt.csv'
blocks_table = 'maxmind_ipv4_geo_blocks'

locations_file_s3 = 's3://hc-interim-data/GeoIP2-City-Locations-en.csv'
locations_table = 'maxmind_geo_locations'


s3_access_role = ''

blocks_drop = DROP.format(schema=schema, table=blocks_table)
locations_drop = DROP.format(schema=schema, table=locations_table)

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

copy = """
    COPY {schema}.{table} 
    FROM '{s3_location}'
    iam_role '{iam_role}'
    CSV IGNOREHEADER AS 1
    ;
"""

blocks_copy = copy.format(schema=schema, table=blocks_table, s3_location=blocks_file_s3, iam_role=s3_access_role)
locations_copy = copy.format(schema=schema, table=blocks_table, s3_location=blocks_file_s3, iam_role=s3_access_role)

if __name__ == '__main__':

    sql_commands = [
        blocks_drop,
        locations_drop,
        blocks_create,
        locations_create,
        blocks_copy,
        locations_copy
    ]

    with RedshiftContextHc as db_context:
        for command in sql_commands:
            db_context.exec(command)

