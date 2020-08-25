.PHONY: download notebook test ship update

download:
	python cpi/download.py


notebook:
	runipy notebooks/analysis.ipynb

test:
	flake8 cpi
	coverage run tests.py
	coverage report -m


ship:
	rm -rf build/
	python setup.py sdist bdist_wheel
	twine upload dist/* --skip-existing


update:
	python cpi/download.py
	runipy notebooks/analysis.ipynb
	python mkdocs.py
