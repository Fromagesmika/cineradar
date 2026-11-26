# Hugging Face Space for CinéRadar. Put this file and README.md at the root of a Docker Space:
# at each build it takes the code from GitHub, so the Space only needs a rebuild after a push
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user

ARG REPO=Fromagesmika/cineradar
ARG BRANCH=main
# this file changes with every new commit on the branch, so a rebuild always takes the latest code
ADD https://github.com/${REPO}/commits/${BRANCH}.atom /tmp/latest-commits.atom
RUN git clone --depth 1 --branch ${BRANCH} https://github.com/${REPO}.git app
WORKDIR /home/user/app
RUN pip install --no-cache-dir --user -r requirements.txt

EXPOSE 7860
CMD ["bash", "deploy/huggingface/run.sh"]
