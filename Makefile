IMAGE_NAME := akkadeeemikk/mats
CONTAINER_NAME := research

build_mats:
	docker build -f ./Dockerfile -t $(IMAGE_NAME) .

stop:
	docker stop $(CONTAINER_NAME)

jupyter:
	jupyter lab --allow-root --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token=mats

run_docker:
	docker run -it --rm \
		--ipc=host \
		--network=host \
		--gpus=all \
		-v ./:/workspace/ \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME) bash
