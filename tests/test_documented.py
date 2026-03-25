from pathlib import Path
from experimaestro.tools.documentation import DocumentationAnalyzer


def test_documented():
    """Test if every configuration is documented"""
    doc_path = Path(__file__).parents[1] / "docs" / "source" / "index.rst"
    # Exclude modules that depend on xpmir (should be moved there eventually)
    analyzer = DocumentationAnalyzer(
        doc_path,
        ["xpm_torch"],
        {
            "xpm_torch.test",
        },
    )

    analyzer.analyze()
    analyzer.report()
    analyzer.assert_valid_documentation()
