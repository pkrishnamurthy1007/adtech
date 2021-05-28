from pkg_resources import resource_filename as rscfn
import os
from ds_utils.db.connectors import HealthcareDW,PivotDW
TABLE_DUMPS = rscfn(__name__,"TABLE_DUMPS")
os.makedirs(TABLE_DUMPS,exist_ok=True)
def fetch_head(rsc,limit=1000,src=HealthcareDW):
    with src() as db:
        db \
            .to_df(f"select * from {rsc} limit {limit}") \
            .to_csv(f"{TABLE_DUMPS}/{src.__name__}.{rsc}.csv")
    return db