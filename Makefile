.PHONY: help clean clean-pyc clean-build list test test-all coverage docs release sdist

help:
	@echo "clean-build - remove build artifacts"
	@echo "clean-pyc - remove Python file artifacts"
	@echo "lint - check style with flake8"
	@echo "test - run tests quickly with the default Python"
	@echo "coverage - check code coverage quickly with the default Python"
	@echo "docs - generate Sphinx HTML documentation, including API docs"

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
	open docs/_build/html/index.html

lint:
	py.test --junit-xml=test_output/flake8/results.xml --flake8 -p no:regtest --cache-clear uncoverml

test:
	py.test --junit-xml=test_output/pytest/results.xml --cache-clear --disable-pytest-warnings

coverage:
	py.test --junit-xml=test_output/pytest/results.xml --cov=uncoverml --cov-report=html:test_output/coverage --cache-clear --cov-fail-under=30 ./tests --disable-pytest-warnings

