update-submodules:
	git submodule update --recursive --remote

setup:
	pip install -r requirements.txt

smi:
	nvidia-smi -l 3

tb:
	tensorboard --logdir results

format:
	find . -name "*.py" | xargs isort
	black -t py36 .
	mypy --ignore-missing-imports .

debug: format
	python src/train.py --config config/debug_mug.yml

build:
	docker build . -f Dockerfile.cpu -t dcvgan.cpu

test:
	docker run --rm --name dcvgan.cpu dcvgan.cpu \
		python -m unittest discover -s src -p '*_test.py'

