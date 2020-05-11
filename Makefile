.PHONY: help clean clean-pyc clean-build lint test coverage docs ghp dist release

help:
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "lint - check style with pylint"
	@echo "test - run tests quickly with the default Python"
	@echo "coverage - check code coverage"
	@echo "docs - generate Sphinx HTML documentation, including API docs"
	@echo "ghp - upload docs to github pages"
	@echo "dist - build source and wheel distributions"
	@echo "release - upload source and wheel distributions to PyPi."

clean: clean-build clean-pyc

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +

docs:
	rm -f docs/uncoverml.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ uncoverml
	$(MAKE) -C docs clean
	$(MAKE) -C docs html

ghp: docs
	$(MAKE) -C docs ghp

lint:
	pylint uncoverml

test:
	pytest --disable-warnings ./tests 

coverage:
	pytest --disable-warnings --cov=uncoverml --cache-clear --cov-fail-under=30 ./tests 

partition_test:
	pytest --disable-warnings ./tests/test_scripts.py --num_parts 2

processor_test:
	pytest --disable-warnings ./tests/test_scripts.py --num_procs 2

hardware_test: partition_test processor_test
	pytest --disable-warnings ./tests/test_scripts.py --num_parts 2 --num_procs 2

dist: clean
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

release-test: dist
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

release: dist
	twine upload dist/*
