.PHONY: install data check all falsify sensitivity test clean

install:
	pip install -r requirements.txt

data:
	python scripts/download_data.py

check:
	python scripts/01_cohorts.py

falsify:
	python scripts/04_falsification.py

all:
	python scripts/run_all.py

sensitivity:
	@echo "run repeatedly until it prints DONE"
	python scripts/05_sensitivity.py 300

test:
	pytest -q || python tests/run_tests.py

clean:
	rm -rf results/csv/*.csv results/figures/* results/tables/* .sobol_cache
