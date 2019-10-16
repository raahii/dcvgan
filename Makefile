setup:
	pip install -r requirements.txt

format:
	find . -name "*.py" | xargs isort
	black -t py36 .
	mypy --ignore-missing-imports .

debug: format
	python src/train.py --config config/debug_mug.yml

smi:
	nvidia-smi -l 3

tb:
	tensorboard --logdir results

build:
	docker build . -f Dockerfile.cpu -t dcvgan.cpu

test:
	docker run --rm --name dcvgan.cpu dcvgan.cpu \
		python src/models_test.py && \
		python src/utils_test.py
