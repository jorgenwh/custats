.PHONY: all install install-deps uninstall clean

all: install

install: clean
	pip install .

install-deps:
	pip install -r requirements.txt

uninstall: clean
	pip uninstall custats

clean:
	$(RM) -rf build custats.egg-info
