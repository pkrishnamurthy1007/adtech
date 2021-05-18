import os
import base64
import json
import boto3
ADTECH_ENV_SECRET = "SM_ENV_BASE"

def dump_env_to_aws():
    print(f"Attempting to write environ to secret: `{ADTECH_ENV_SECRET}`")

    secretsmanager = boto3.client('secretsmanager')
    
    environ_bytes = \
        base64.b64encode(
            json.dumps(
                {**os.environ}).encode('utf-8'))
    try:
        resp = secretsmanager.create_secret(
            Name=ADTECH_ENV_SECRET,
            SecretBinary=environ_bytes,
        )
    except Exception as ex:
        print(f"Creating secret failed w/ {type(ex)}: {ex} - attempting update")
        resp = secretsmanager.put_secret_value(
            SecretId=ADTECH_ENV_SECRET,
            SecretBinary=environ_bytes,
        )
    
    print("...Success!!")

def load_env_from_aws():
    print(f"Attempting to load environ from: `{ADTECH_ENV_SECRET}`")

    secretsmanager = boto3.client('secretsmanager')
    sm_env_base_secret = secretsmanager.get_secret_value(
        SecretId=ADTECH_ENV_SECRET)
    sm_env_base = json.loads(base64.b64decode(sm_env_base_secret["SecretBinary"]))
    # already set os env vars take precendence over aws vals
    os.environ.update({**sm_env_base, **os.environ})

    print("...Success!!")