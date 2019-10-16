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
	black -t py36 . --exclude "data" --exclude "result" --exclude ".mypy_cache"
	mypy --ignore-missing-imports .

debug: format
	python src/train.py --config config/debug_mug.yml

build:
	docker build . -f Dockerfile.cpu -t dcvgan.cpu

test:
	docker run --rm --name dcvgan.cpu dcvgan.cpu \
		python -m unittest discover -s src -p '*_test.py'

deploy:
	rsync -auvz \
				--delete \
				--exclude='.DS_Store' \
				--exclude='.git*' \
				--exclude='.python-version' \
				--exclude='__pycache__' \
				--exclude='evaluation/models/weights/*' \
				--exclude='data' \
				--exclude='result' \
				--exclude='deploy.sh' \
				--exclude='.mypy_cache' \
				$(shell ghq root)/github.com/raahii/dcvgan/ labo:~/study/dcvgan/
