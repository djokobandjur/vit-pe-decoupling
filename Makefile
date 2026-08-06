.PHONY: verify reproduce verify-reproduction manuscript test clean

verify:
	python scripts/verify_repository.py

reproduce:
	python scripts/reproduce_all.py

verify-reproduction:
	python scripts/verify_reproduction.py

manuscript:
	python scripts/build_manuscript.py

test:
	pytest -q

clean:
	rm -rf artifacts/reproduced build .pytest_cache
