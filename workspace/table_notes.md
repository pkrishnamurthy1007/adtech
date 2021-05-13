hc_raw,hc_disposition
hc_raw,healthcare_clicks_pivot
internal_raw,ph_session_detail
internal_raw,ph_transactional_application
internal_raw,session_enroll

ph_transactional,application

salesforce,accounts
salesforce,opportunities
salesforce,policies

session,application_touch_points
session,master_table
session,paid_media_conversions

tracking,app_premium
tracking,app_premium_ltv
tracking,app_premium_revised


ca79008da23e4ed4b533363dab0b7d92
00007F9E11F8463B9454C87F4DE3A084

"""
@obed
yea, in ph_transactional.application we store sells for different agents and agencies

and that's the source for session_enroll

in sales_center.sales we have exclusive sales made by the sales center

@amal
hmmm - so we have data for:
1. sales of PH plans made by internal and external sales centers in session_enroll and application
2. sales of PH and external plans made by internal sales center in sales_center.sales

but do we have a table indicating sales of external policies made by external sales centers?
(but linked to a pivot health session?)
 => yes - this is overflow process - pay comission in case of sale
 => `sales_center.call_performance` maps to overflow data

@obed
I would recomment to use session.user_attribution_model
instead of session_enroll

@amal
so - is user_id unique per session_id? and does that mean that there may be many user ids and sessions per real person?
9:21
should I be merging these based on email or phone #?
9:23
I also noticed that the user_attribution_table is much smaller than session_enroll (2M vs 20M) - do you know whats causing this discrepancy?
is it sessions being aggregated to a phone # / email?

@obed
Tha happend in the past, but now we enforce one email per user_id
when the user inputs the email in the website we could change the user id according the email that was input
in user attribution table - one row per person per policy bought

for that we have a table called
9:10
sales_center.multicarrier_sales
9:10
and
9:10
sales_center.sales
9:10
for all sales consolidation
"""

# taboola attribution
log_upload.tron_session_revenue 
and for PH you could 
reports.sales_view

tron.session_revenue
-  equivalent to user_attribution_table
-  also attributes 