FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-devel

WORKDIR /app/ChatTTS

ADD . /app/ChatTTS

RUN apt-get update \
    && apt-get install -y curl \
    && curl -fsSL https://deb.nodesource.com/setup_lts.x | bash -  \
    && apt-get install -y nodejs \
    && npm install --global yarn  \
    && rm -rf node_modules \
    && yarn config set registry https://registry.npm.taobao.org/ \
    && yarn \
    && yarn build \
    && pip install --no-cache-dir -r requirements.txt \
    && chmod +x ./docker/start.sh

RUN apt-get clean all \
    && rm -rf /var/lib/apt/lists/* 

EXPOSE 3000
EXPOSE 8000

ENTRYPOINT ["bash", "-c"]
CMD ["./docker/start.sh"]
