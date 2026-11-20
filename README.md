# AI-Powered OBE Syllabus Generator

A Python desktop app and CLI for generating validated course outcomes and
weekly learning plans using a local Ollama model.

## Features

- Choose built-in courses or add custom courses.
- Add a custom course description and optionally improve it with AI.
- Generate 4–6 validated Course Learning Outcomes across Knowledge, Skills,
  and Attitude domains.
- Generate a configurable weekly learning plan (default 8 weeks, maximum 12).
- Review cleaned summaries and JSON output in the desktop app.
- Export a selected course summary as a paginated PDF.
- Run offline with the mock provider.

## Requirements

- Python 3.10 or newer
- Tkinter (included with most Windows and macOS Python installers)
- Ollama with a model installed for live generation

Install the Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

For live generation, install Ollama and download a model, for example:

```powershell
ollama pull qwen2.5:7b
```

## Run the desktop app

On Windows, double-click `run_gui.bat`, or run:

```powershell
python gui.py
```

Add courses on **Create batch**, configure generation, and select **Generate
outcomes**. Follow progress in the live log. Results opens when generation
finishes.

## Run from the command line

Run an offline mock batch:

```powershell
python llm_engine.py --mock
```

Generate selected built-in courses:

```powershell
python llm_engine.py --course CS321 --course CS323
```

Generate a custom course with a description and a 12-week plan:

```powershell
python llm_engine.py --course custom --course-title "Animal Welfare" `
  --course-description "Covers animal welfare principles and ethical care." `
  --improve-course-description --weeks 12
```

Use `--course-code` to provide a custom course code. When omitted, the next
available code follows the built-in sequence (currently `CS325`, then `CS326`,
and onward). The older `--subject` and `--subject-title` flags remain
supported as aliases.

## Generation settings

- **Fast**, **Balanced**, and **Quality** trade off call count and audit depth.
- `--weeks` sets one shared schedule length for all courses: 1–12, default 8.
- `--timeout` sets the maximum duration of an individual Ollama call.
- `--debug` prints raw model output and automatic repair details.
- `--mock` runs offline without an Ollama daemon.

## Output

The CLI and desktop app write `sample_validated_output.json` in the project
folder by default. This is a generated runtime file and is ignored by Git.
Use `--output PATH` with the CLI to choose another destination.

## Tests

Run the test suite with Python's built-in unittest runner:

```powershell
python -m unittest discover -s tests -v
```

## Project files

- `gui.py` — Tkinter desktop app
- `llm_engine.py` — Ollama integration, generation, retries, CLI, and mock mode
- `obe_schemas.py` — Pydantic validation models
- `pdf_export.py` — dependency-free course-summary PDF export
- `tests/` — schema, generation, and PDF tests
