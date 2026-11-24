# nbconvert LaTeX templates

## `unicode_latex`

XeLaTeX PDF export with fonts that cover Unicode Greek and similar
symbols in notebook Markdown (α, σ, λ, …). The stock nbconvert latex
template leaves Latin Modern as the default; those glyphs are missing.

Also defines `\newcounter{none}` when the stock template has not, so
pandoc captionless `longtable`s build under current TeX Live.

### Usage

From a directory that contains an executed `.ipynb`:

```bash
jupyter nbconvert --to pdf \
  --template unicode_latex \
  --TemplateExporter.extra_template_basedirs=/path/to/port-notes/nbconvert_templates \
  meijering_alpha.ipynb
```

Or set once in `~/.jupyter/jupyter_nbconvert_config.py`:

```python
c.TemplateExporter.extra_template_basedirs = [
    "/path/to/port-notes/nbconvert_templates"
]
c.LatexExporter.template_name = "unicode_latex"
```

Fonts (macOS): STIX Two Text, STIX Two Math, Menlo.
