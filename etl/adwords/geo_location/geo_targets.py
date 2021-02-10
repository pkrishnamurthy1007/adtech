import csv
import re
import ssl
from codecs import iterdecode
from contextlib import closing
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import pandas as pd
import pendulum as dt
import requests
from ds_utils.db.connectors import AnalyticsDB, HealthcareDW
from ds_utils.db.generic_sql import drop_table, truncate_table
from ds_utils.db.mysql_sql import load_local_sql
from etl.adwords.constant_tables.geo_location.dma_names import DMA_LIST

SCHEMA = 'log_upload'
TABLE = 'adwords_geo_targets'

"""
    This creates a lookup table that contains all of the Google Ads geographic criterion IDs and the asssociated
    meta-data.  These "location IDs" can be joined to all of google's performance, including the click level report,
    so that google's notion of every user location can be known to us.  These IDs can also be used by account
    managers for implementing location-based adjustments in the Google Ads UI.
        - Scrapes the google developer doc page for the path to the latest file
        - Downloads the file and loads into analytics DB.
        - Appends DMA criterion IDs.  These are not provided directly by google, even though they appear in their 
          reporting and can be targeted.  This is because DMA regions are copywrit by Neilson.
        - Some minor intermediary transformations and cleanups to make joins easier
"""

def get_file_link_from_scrape():
    """
    Adwords only posts these files in their api documentation.  Scrapes the page with regex
    to find the most recent file download path
    """
    domain = 'https://developers.google.com'
    scrape_path = domain + '/adwords/api/docs/appendix/geotargeting'
    file_link_match = '<a href="(/adwords/api/docs/appendix/geoip/geotargets-\d{4}-\d{2}-\d{2}.csv)">'

    with urlopen(scrape_path) as response:
        geotargets_html = response.read()

    file_download_links = re.findall(file_link_match, str(geotargets_html))

    latest_date, latest_link = None, None
    for link in file_download_links:
        file_date = dt.parse(re.findall('-(\d{4}-\d{2}-\d{2}).csv', link)[0])

        if latest_date is None:
            latest_date = file_date
            latest_link = link
        else:
            if file_date > latest_date:
                latest_date = file_date
                latest_link = link

    latest_csv_link = domain + latest_link

    return latest_date, latest_csv_link


def setup_tables_mysql():
    drop = drop_table(SCHEMA, TABLE, if_exists=True)

    create = f"""
        CREATE TABLE {SCHEMA}.{TABLE} ( 
                location_id BIGINT PRIMARY KEY, 
                name VARCHAR(128), 
                canonical_name VARCHAR(512), 
                parent_id BIGINT, 
                country_code VARCHAR(8), 
                target_type VARCHAR(128), 
                status VARCHAR(8),
                file_version DATE DEFAULT null
            )
        ;
    """
    with AnalyticsDB() as db_context:
        db_context.exec(drop)
        db_context.exec(create)


def get_loaded_version():

    select = f"SELECT coalesce(max(file_version), cast('2000-01-01' AS date)) FROM {SCHEMA}.{TABLE};"

    with AnalyticsDB() as db_context:
        res = db_context.fetch(select)[0][0]

    return dt.datetime(res.year, res.month, res.day)


def load_latest_csv(url, version):

    truncate = truncate_table(SCHEMA, TABLE, if_exists=True)

    with NamedTemporaryFile(mode='w') as load_file:
        writer = csv.writer(load_file, 'unix')

        # stream request
        with closing(requests.get(url, stream=True)) as r:
            reader = csv.reader(iterdecode(r.iter_lines(), 'utf-8'))
            # skip header row
            next(reader)
            # write to tempfile
            for row in reader:
                writer.writerow(row + [version.to_datetime_string()])

        load_file.flush()

        load_sql = load_local_sql(SCHEMA, TABLE, filepath=load_file.name)
        # to debug load statement
        # print(r'%s' %load_sql)

        with AnalyticsDB() as db_context:
            db_context.exec(truncate)
            db_context.exec(load_sql)

def load_dmas():

    metro_states_query = """
        SELECT DISTINCT
            metro_code::INT AS dma_code,
            '(' || listagg(state_code, ',') OVER (PARTITION BY metro_code) || ')' state_list
        FROM (
            SELECT DISTINCT
                metro_code,
                subdivision_1_iso_code AS state_code
            FROM data_science.maxmind_geo_locations
            WHERE country_iso_code = 'US'
                AND metro_code != ''
            ) AS sub
        ;
    """

    with HealthcareDW() as db_context:
        df = db_context.to_df(metro_states_query)

    dma_df = pd.DataFrame(DMA_LIST, columns=['dma_code', 'name']).merge(df, how='inner', on='dma_code')

    dma_df['canonical_name'] = dma_df['name'] + ', ' + dma_df['state_list'] + ', ' + 'United States'
    dma_df['parent_id'] = 2840
    dma_df['location_id'] = ('200' + dma_df['dma_code'].astype(str)).astype(int)
    dma_df['country_code'] = 'US'
    dma_df['status'] = 'Active'
    dma_df['target_type'] = 'DMA'

    dma_df = dma_df[['location_id', 'name', 'canonical_name', 'parent_id', 'country_code', 'target_type', 'status']]

    table_inserts = dma_df.values.tolist()
    table_inserts.append([0,'location unknown', '\\N', '\\N', '\\N', '\\N', 'Active' ])

    with NamedTemporaryFile(mode='w') as load_file:
        writer = csv.writer(load_file, 'unix')
        for row in table_inserts:
            writer.writerow(row)

        load_sql = load_local_sql(SCHEMA, TABLE, filepath=load_file.name)
        # to debug load statement
        # print(r'%s' %load_sql)

        with AnalyticsDB() as db_context:
            db_context.exec(load_sql)

if __name__== '__main__':

    ssl._create_default_https_context = ssl._create_unverified_context

    # uncomment to wipe table and start clean or for first time run
    #setup_tables_mysql()

    file_version, file_download_link = get_file_link_from_scrape()

    if file_version > get_loaded_version():
        load_latest_csv(file_download_link, file_version)
        load_dmas()

