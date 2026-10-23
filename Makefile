# Sift: one-command demo. `make demo` builds the UI, starts the CPU-only backend and the frontend, prints the URL.
PY      ?= .venv/bin/python
API_PORT ?= 8000
UI_PORT  ?= 5173
FE       = app/frontend

.PHONY: demo frontend backend stop e2e

frontend:
	cd $(FE) && ([ -d node_modules ] || npm ci) && npm run build

backend:
	CUDA_VISIBLE_DEVICES='' SIFT_DEVICE=cpu PYTHONPATH=src .venv/bin/uvicorn app.backend.main:app --port $(API_PORT)

demo: frontend
	@mkdir -p .run
	@echo "Starting backend (CPU only) on :$(API_PORT) ..."
	@CUDA_VISIBLE_DEVICES='' SIFT_DEVICE=cpu PYTHONPATH=src nohup .venv/bin/uvicorn app.backend.main:app --port $(API_PORT) > .run/backend.log 2>&1 & echo $$! > .run/backend.pid
	@nohup npm --prefix $(FE) run preview -- --port $(UI_PORT) > .run/frontend.log 2>&1 & echo $$! > .run/frontend.pid
	@for i in $$(seq 1 40); do curl -sf localhost:$(API_PORT)/api/health >/dev/null && break; sleep 0.5; done
	@echo ""
	@echo "Product:            http://localhost:$(UI_PORT)/"
	@echo "Workspace:          http://localhost:$(UI_PORT)/app"
	@echo "Developer console:  http://localhost:$(UI_PORT)/dev"
	@echo "API docs:           http://localhost:$(API_PORT)/docs"
	@echo "Stop with: make stop    Logs in .run/"

stop:
	-@kill $$(cat .run/backend.pid) $$(cat .run/frontend.pid) 2>/dev/null; rm -f .run/*.pid

e2e:
	CUDA_VISIBLE_DEVICES="" SIFT_DEVICE=cpu PYTHONPATH=src $(PY) -m pytest e2e -q -p no:cacheprovider

.PHONY: reproduce
reproduce:
	PYTHONPATH=src .venv/bin/python experiments/run_baselines.py
	PYTHONPATH=src .venv/bin/python experiments/make_tables.py
	@echo "Heavy GPU runs: see experiments/run_all_gpu.sh and experiments/run_baselines_gpu.sh (hours; not re-run here)"
