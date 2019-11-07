update-submodules:
	git submodule update --recursive --remote

setup:
	pip install -r requirements.txt

docker-release:

smi:
	nvidia-smi -l 3

tb:
	tensorboard --logdir result --host 0.0.0.0

format:
	find . -name "*.py" | xargs isort
	black -t py36 . --exclude "data" --exclude "result" --exclude ".mypy_cache"
	mypy --ignore-missing-imports .

debug:
	python src/train.py --config config/debug-isogd-optical-flow.yml

test:
	python -m unittest discover -s src/test -p 'test_*.py'

build:
	docker build . -f docker/Dockerfile.cpu -t raahii/dcvgan:test

ci:
	docker run --volume ${PWD}:/home/user/dcvgan --rm raahii/dcvgan:test make test

deploy:
	rsync -auvz \
	      --delete \
	      --exclude-from=.rsyncignore \
	      $(shell ghq root)/github.com/raahii/dcvgan/ labo:~/study/dcvgan/
