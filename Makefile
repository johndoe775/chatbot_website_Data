env:
	python3 -m venv .venv 

activate:
	run . .venv/bin/activate

install:
	python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt

git:
	git add .
	git status
	git commit -m "recent edits"
	git push

format:
	python3 -m black . --include '\.py'

run:
	streamlit run app1.py