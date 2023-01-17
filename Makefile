SRC := $(shell find src/ -name "*.py")

format:
	@autoflake --remove-all-unused-imports -i $(SRC)
	@isort $(SRC)
	@black --line-length 100 $(SRC)

typecheck:
	mypy src/

clean:
	rm -rf **/__pycache__
	rm -rf **/.ipynb_checkpoints
	rm -rf .mypy_cache
	rm -rf .hypothesis
	rm -rf .pytest_cache
