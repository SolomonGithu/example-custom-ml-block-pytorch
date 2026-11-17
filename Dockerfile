# Simple Ubuntu 24.04 base image with Python3.12 and CUDA setup already (for GPU training)
FROM edge-impulse-pub-ml-block-base.dockermirror.stablebuild.com/public.ecr.aws/g7a8t7v6/ei-custom-ml-block-base:v1.95.10

COPY requirements_torch_2.13.0_cuda.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --break-system-packages --no-deps -r requirements_torch_2.13.0_cuda.txt

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --break-system-packages --default-timeout=600 --retries 10 -r requirements.txt 
    # Updated to make pip be patient

# Ultralytics, installed without letting it replace the pinned torch
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --break-system-packages --default-timeout=600 --retries 10 --no-deps --extra-index-url https://download.pytorch.org/whl/cu129 \
      ultralytics ultralytics-thop torchvision==0.28.0 \
 && pip3 install --break-system-packages matplotlib opencv-python-headless pillow pyyaml requests scipy polars psutil tqdm

# Bake the pretrained weights in (the block has no internet inside Edge Impulse)
RUN mkdir -p /opt/weights && cd /opt/weights \
 && python3 -c "from ultralytics import YOLO; [YOLO(f'yolo26{s}.pt') for s in 'nsm']" \
 && ls -la /opt/weights

COPY . ./

ENTRYPOINT ["python3", "-u", "train.py"]