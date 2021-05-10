# Boilerplate Media Alpha API Code

While, Media Alpha's API is very easy to use and doesn't require a lot of boiler plate code,
we aim to still have it be separate.

## Examples

```py
from api.mediaalpha.mediaalpha_client import MediaAlphaAPIClient
token = os.getenv("MEDIAALPHA_TOKEN")
client = MediaAlphaAPIClient(base_url = "https://insurance-api.mediaalpha.com/220", token=token)
r = client.set_time_of_day_modifiers(schedule_payload, campaign = CAMPAIGN_NUMBER)
# Status<200>
```
