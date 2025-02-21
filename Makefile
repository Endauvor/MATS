IMAGE_NAME := explosion
CONTAINER_NAME := detnet
DOCKER_TAG := latest
PURE_TAG := pure
TRT_TAG := trt

build_pure:
	docker build -f docker/pure/Dockerfile -t $(IMAGE_NAME)_$(PURE_TAG) .

build_trt:
	docker build -f docker/compression/Dockerfile -t $(IMAGE_NAME)_$(TRT_TAG) .

run_docker_pure_bash:
	docker run -it \
		--ipc=host \
  		--network=host \
  		--gpus=all \
  		-v ./:/detector/ \
  		--name $(CONTAINER_NAME)_$(PURE_TAG) \
  		$(IMAGE_NAME)_$(PURE_TAG) bash

run_docker_trt_bash:
	docker run -it \
		--ipc=host \
  		--network=host \
  		--gpus=all \
  		-v ./:/detector/ \
  		--name $(CONTAINER_NAME)_$(TRT_TAG) \
  		$(IMAGE_NAME)_$(TRT_TAG) bash

load_dataset:
	mkdir ../datasets && \
	wget https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/download/1.1/dataset_birdview_vehicles.zip && \
	unzip -d ../datasets/ /detector/tutorial/dataset_birdview_vehicles.zip

load_weights:
	mkdir weights && \
	wget https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/download/1.0/efficientdet-d0.pth -O weights/efficientdet-d0.pth


stop:
	docker stop $(CONTAINER_NAME)

jupyter:
	jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token=msuai

build:
	docker build -f Dockerfile -t $(IMAGE_NAME)_$(PURE_TAG) .

run_docker_pure_bash:
	docker run -it \
		--ipc=host \
  		--network=host \
  		--gpus=all \
  		-v ./:/app/ \
  		--name $(CONTAINER_NAME) \
  		$(IMAGE_NAME) bash
