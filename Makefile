IMAGE ?= chatbot:dev
PORT ?= 8000
ENV_FILE ?= .env

.PHONY: build run rebuild shell

build:
	docker build -t $(IMAGE) .

rebuild:
	docker build --no-cache -t $(IMAGE) .

run:
	docker run --rm -p $(PORT):8000 --env-file $(ENV_FILE) $(IMAGE)

shell:
	docker run --rm -it -p $(PORT):8000 --env-file $(ENV_FILE) $(IMAGE) /bin/sh
