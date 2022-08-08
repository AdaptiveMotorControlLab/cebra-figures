
targets := $(patsubst src/%.py,figures/%.ipynb,$(wildcard src/*.py))

requirements:
	echo install

all: clean $(targets)

build/%.ipynb:
	echo $% $@
	mkdir -p build
	python -m jupytext src/$*.py --from py --to ipynb --output build/$*.ipynb

figures/%.ipynb: requirements build/%.ipynb
	echo build/$*.ipynb $@
	python -m jupyter nbconvert --to notebook --execute build/$*.ipynb --output $*_tmp.ipynb
	mv build/$*_tmp.ipynb $@

.PHONY: clean
