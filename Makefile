
targets := $(patsubst src/%.py,figures/%.ipynb,$(wildcard src/*.py))

requirements:
	pip install -r requirements.txt

all: $(targets)

html:
	PYTHONPATH=.:third_party/ sphinx-build -M html . _build

build/%.ipynb:
	echo $% $@
	mkdir -p build
	python -m jupytext src/$*.py --from py --to ipynb --output build/$*.ipynb

figures/%.ipynb: build/%.ipynb
	echo build/$*.ipynb $@
	python -m jupyter nbconvert --ExecutePreprocessor.kernel_name=python3 --to notebook --execute build/$*.ipynb --output $*_tmp.ipynb
	mv build/$*_tmp.ipynb $@

.PHONY: clean
