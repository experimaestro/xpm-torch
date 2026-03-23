from pathlib import Path
from experimaestro.tools.documentation import DocumentationAnalyzer


def test_documented():
    """Test if every configuration is documented"""
    doc_path = Path(__file__).parents[1] / "docs" / "source" / "index.rst"
    analyzer = DocumentationAnalyzer(doc_path, ["xpm_torch"], {"xpm_torch.test"})

    analyzer.analyze()
    analyzer.report()
