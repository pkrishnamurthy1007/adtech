FROM continuumio/anaconda3
ARG GIT_SSH_KEY

ADD . /app

RUN \
    echo "${GIT_SSH_KEY}\n" > GIT_SSH_KEY && \
    chmod 600 GIT_SSH_KEY && \
    mkdir -p /root/.ssh/ && \
    ssh-keyscan -t rsa github.com > ~/.ssh/known_hosts && \
    GIT_SSH_COMMAND="ssh -i GIT_SSH_KEY" pip install -r /app/requirements.txt && \
    rm -rf /root/.cache
RUN apt-get -y purge --auto-remove git

ENTRYPOINT ["python","/app/bing_keyword_bidder.py"]
