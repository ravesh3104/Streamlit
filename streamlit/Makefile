.PHONY: help install check data train report notebook app all clean test

help:
	@echo "Industrial Equipment Failure Prediction"
	@echo ""
	@echo "  make install    install python dependencies"
	@echo "  make check      verify your environment (run this first)"
	@echo "  make data       (re)download the raw AI4I 2020 dataset"
	@echo "  make train      clean -> train -> evaluate -> explain -> save artifacts"
	@echo "  make report     regenerate reports/ (.md + .docx) from results.json"
	@echo "  make notebook   rebuild and execute the walkthrough notebook"
	@echo "  make app        launch the Streamlit dashboard"
	@echo "  make all        data + train + report + notebook"
	@echo "  make clean      remove generated artifacts"

install:
	pip install -r requirements.txt

check:
	python scripts/check_env.py

data:
	python -m src.download

train:
	python -m src.train

report:
	python scripts/build_report.py

notebook:
	python scripts/build_notebook.py
	jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=900 \
		notebooks/01_failure_prediction_walkthrough.ipynb

app:
	streamlit run app/streamlit_app.py

all: data train report notebook
	@echo ""
	@echo "Built. Launch the dashboard with:  make app"

clean:
	rm -rf artifacts/*.joblib artifacts/*.json artifacts/*.csv artifacts/figures/*.png
	rm -rf reports/figures reports/*.docx reports/*.md
	rm -rf data/clean_ai4i2020.csv data/synthetic_corrupted_ai4i.csv
	@echo "cleaned (data/raw_ai4i2020.csv is preserved)"
