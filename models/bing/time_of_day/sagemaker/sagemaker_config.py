#%%
import boto3
import sys
import sagemaker
import numpy as np
from sagemaker import get_execution_role

REGION = "us-east-1"
DEFAULT_PROFILE = "default"
TREVOR_PROFILE = "trevor"
# SM_STUDIO_PROFILE = "ph-data-science-sagemaker-studio"
# used for all sagemaker studio projects
SM_HC_PROFILE = "hc-sagemaker-default-execution-role"
# boto3.setup_default_session(profile_name=SM_STUDIO_PROFILE)
boto_sess = boto3.session.Session(
    profile_name=SM_HC_PROFILE,
    region_name=REGION,
)

sm_sess = sagemaker.Session(boto_sess)
# ??? maybe?
# role = "arn:aws:iam::915124832670:role/ph-data-science-sagemaker-studio-role"
role = get_execution_role(sagemaker_session=sm_sess)
region = boto_sess.region_name
account_id = boto_sess.client("sts").get_caller_identity().get('Account')
s3_output = sm_sess.default_bucket()
#%%
ORG = "adtech"
PROJECT = "bing_tod_modifiers"
s3_prefix = f'sagemaker/{ORG}/{PROJECT}'
#%%
from sagemaker.analytics import ExperimentAnalytics
from smexperiments.experiment import Experiment
from smexperiments.trial import Trial
from smexperiments.trial_component import TrialComponent
from smexperiments.tracker import Tracker
import time

bing_tod_experiment = Experiment.create(
    experiment_name=f"{ORG}-{PROJECT}-{int(time.time())}",
    description="Purchase intent prediction with lightGBM",
    sagemaker_boto_client=boto_sess.client('sagemaker'))
print(bing_tod_experiment)
#%%
DOCKER_PROCESS_CONFIG_DIR = "docker_process"
!mkdir -p $DOCKER_PROCESS_CONFIG_DIR
!ls
# %%
# %%writefile docker-proc-evaluate/Dockerfile
# FROM continuumio/anaconda3:latest
# # need to figure out how to use build args here
# RUN git clone git@github.com:healthcarecom/datascience-utils.git && \
#     pip install -r datascience-utils/requirements.txt && \
#     pip install datascience-utils
# ENV PYTHONUNBUFFERED=TRUE
# ENTRYPOINT ["python3"]
#%%
# %%
ecr_repository = f'sagemaker-{ORG}-{PROJECT}'
tag = ':latest'
uri_suffix = 'amazonaws.com'
processing_repository_uri = f'{account_id}.dkr.ecr.{region}.{uri_suffix}/{ecr_repository + tag}'
#%%
%%writefile $DOCKER_PROCESS_CONFIG_DIR/Dockerfile_test
FROM continuumio/anaconda3:latest
COPY . /app
RUN pip install -r app/datascience-utils/requirements.txt && \
    pip install app/datascience-utils
ENV PYTHONUNBUFFERED=TRUE
ENTRYPOINT ["python3","app/tod_dp.py"]
#%%
!cp tod_dp.py $DOCKER_PROCESS_CONFIG_DIR/tod_dp.py
# %%
# Create ECR repository and push docker image
!docker build --no-cache \
    -t tod_dp_test \
    $DOCKER_PROCESS_CONFIG_DIR \
    -f $DOCKER_PROCESS_CONFIG_DIR/Dockerfile-test
!docker run tod_dp_test
#%%
%%writefile $DOCKER_PROCESS_CONFIG_DIR/Dockerfile
FROM continuumio/anaconda3:latest
COPY . /app
RUN pip install -r app/datascience-utils/requirements.txt && \
    pip install app/datascience-utils
ENV PYTHONUNBUFFERED=TRUE
ENTRYPOINT ["python3"]
#%%
!cat $DOCKER_CONFIG_DIR/Dockerfile
#%%
!git clone \
    git@github.com:healthcarecom/datascience-utils.git \
    $DOCKER_PROCESS_CONFIG_DIR/datascience-utils
# %%
# Create ECR repository and push docker image
!docker build \
    -t $ecr_repository \
    $DOCKER_PROCESS_CONFIG_DIR
# !$(aws ecr get-login --region $region --registry-ids $account_id --no-include-email)
login_cmds = !aws ecr get-login --region $region --profile $SM_HC_PROFILE --no-include-email
docker_login_cmd = login_cmds[0]
!$docker_login_cmd
!aws ecr create-repository --repository-name $ecr_repository
!docker tag {ecr_repository + tag} $processing_repository_uri
!docker push $processing_repository_uri
#%%
from sagemaker.processing import ScriptProcessor
script_processor = ScriptProcessor(command=['python3'],
                                   image_uri=processing_repository_uri,
                                   role=role,
                                   instance_count=1,
                                   instance_type='ml.c5.xlarge')
# %%
processing_job_name
#%%
from sagemaker.processing import ProcessingInput, ProcessingOutput
from time import gmtime, strftime 

GMT_DHMS = strftime("%d-%H-%M-%S", gmtime())
processing_job_name = f"{ORG}-{PROJECT}-{GMT_DHMS}"
processing_job_name = processing_job_name.replace("_","-")
output_destination = f's3://{s3_output}/{s3_prefix}/data'

script_processor.run(
    code='tod_dp.py',
    job_name=processing_job_name,
    inputs=[
    #   ProcessingInput(
    #     source=raw_s3,
    #     destination='/opt/ml/processing/input')
    ],
    outputs=[
        # ProcessingOutput(
        #     output_name='train',               
        #     destination='{}/train'.format(output_destination),
        #     source='/opt/ml/processing/train'),
        # ProcessingOutput(
        #     output_name='test',
        #     destination='{}/test'.format(output_destination),
        #     source='/opt/ml/processing/test')
    ],
    experiment_config={
        "ExperimentName": bing_tod_experiment.experiment_name,
        "TrialComponentDisplayName": "Processing",
    }
)
#%%
preprocessing_job_description = script_processor.jobs[-1].describe()
preprocessing_job_description
# %%
output_config = preprocessing_job_description['ProcessingOutputConfig']
for output in output_config['Outputs']:
    if output['OutputName'] == 'train':
        preprocessed_training_data = output['S3Output']['S3Uri']
        print(preprocessed_training_data)
    if output['OutputName'] == 'test':
        preprocessed_test_data = output['S3Output']['S3Uri']
        print(preprocessed_test_data)

# %%
for trial in bing_tod_experiment.list_trials():
    proc_job = trial
    break

lightgbm_tracker = Tracker.load(proc_job.trial_name)
preprocessing_trial_component = lightgbm_tracker.trial_component
print(preprocessing_trial_component)
# %%
