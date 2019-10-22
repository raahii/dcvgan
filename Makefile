update-submodules:
	git submodule update --recursive --remote

setup:
	pip install -r requirements.txt

docker-release:

smi:
	nvidia-smi -l 3

tb:
	tensorboard --logdir results --host 0.0.0.0

format:
	find . -name "*.py" | xargs isort
	black -t py36 . --exclude "data" --exclude "result" --exclude ".mypy_cache"
	mypy --ignore-missing-imports .

debug:
	python src/train.py --config config/debug_mug.yml

build:
	docker build . -f Dockerfile.cpu -t dcvgan.cpu

test:
	docker run --rm --name dcvgan.cpu dcvgan.cpu \
		python -m unittest discover -s src -p '*_test.py'

deploy:
	rsync -auvz \
				--delete \
				--exclude-from=.rsyncignore \
				$(shell ghq root)/github.com/raahii/dcvgan/ labo:~/study/dcvgan/
