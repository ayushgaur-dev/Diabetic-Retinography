"""Reporting pipeline: input dict -> authoritative report dict + renderings."""

from src.reporting.config import load_reporting_config
from src.reporting.deterministic_generator import generate
from src.reporting.html_renderer import render_html
from src.reporting.json_renderer import render_json
from src.reporting.markdown_renderer import render_markdown
from src.reporting.pdf_renderer import render_pdf


def generate_report(input_dict, config=None):
    cfg = config or load_reporting_config()
    return generate(input_dict, cfg).to_dict()


def render_all(report_dict, out_dir, basename="report", figure_path=None):
    """Write JSON/Markdown/HTML (+PDF) renderings. Returns paths dict."""
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    paths["json"] = str(out / f"{basename}.json")
    Path(paths["json"]).write_text(render_json(report_dict), encoding="utf-8")
    paths["md"] = str(out / f"{basename}.md")
    Path(paths["md"]).write_text(render_markdown(report_dict), encoding="utf-8")
    paths["html"] = str(out / f"{basename}.html")
    Path(paths["html"]).write_text(render_html(report_dict, figure_path),
                                   encoding="utf-8")
    paths["pdf"] = str(out / f"{basename}.pdf")
    render_pdf(report_dict, paths["pdf"], figure_path)
    return paths
