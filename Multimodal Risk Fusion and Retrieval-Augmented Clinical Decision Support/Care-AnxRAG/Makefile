PYTHON ?= python

.PHONY: install install-core test coverage validate selfcheck demo build serve clean

install:
	$(PYTHON) -m pip install -e ".[production,dev]"

install-core:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

coverage:
	PYTHONPATH=src coverage run --source=care_anxrag -m pytest -q
	coverage report --fail-under=70

validate:
	./scripts/validate.sh

selfcheck:
	PYTHONPATH=src $(PYTHON) -m care_anxrag.cli selfcheck --offline --project-root .

demo:
	PYTHONPATH=src $(PYTHON) scripts/offline_acceptance.py --project-root .

build:
	$(PYTHON) -m build --no-isolation

serve:
	care-anxrag serve --host 127.0.0.1 --port 8000

clean:
	rm -rf build dist src/*.egg-info .pytest_cache .coverage htmlcov validation-report.json SHA256SUMS
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
