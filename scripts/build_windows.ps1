$ErrorActionPreference = "Stop"

python -m pip install -U pip
pip install -e ".[dev]"
pip install pyinstaller

pyinstaller `
  --name cfgflow `
  --onefile `
  --noconsole `
  --paths src `
  scripts\\cfgflow_entry.py

Write-Host "Built dist\\cfgflow.exe"
