FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-runtime as base
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
ADD packages /app/packages
RUN ls /app/packages | xargs -i pip install /app/packages/{} --force-reinstall -i https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://mirrors.aliyun.com/pypi/simple/

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

