.PHONY: install data train report test app all
install:
	pip install -r requirements.txt && pip install -e .
data:
	python -m ids.data
train:
	python -m ids.train
report:
	python -m ids.report
test:
	pytest -q
app:
	streamlit run app/streamlit_app.py
all: install data train report test
