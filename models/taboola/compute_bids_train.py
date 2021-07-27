#%%
from models.taboola.common import *

from notebooks.aduriseti_shared.utils import *
from models.utils import wavg, get_wavg_by, wstd

from models.taboola.utils import *

split_cols = ["state", "device", "keyword"]
rps_df = agg_rps(start_date, end_date, None, traffic_source=TABOOLA,
                 agg_columns=tuple(["campaign_id", *split_cols, "utc_dt"]))
rps_df = translate_taboola_vals(rps_df)
rps_df = rps_df_postprocess(rps_df)
rps_df = rps_df.reset_index()

clusterer = TaboolaRPSEst(clusts=None,enc_min_cnt=10).fit(
    rps_df.set_index([*split_cols, "utc_dt"]), None)

import joblib
joblib.dump(clusterer,MODEL_PTH)
#%%