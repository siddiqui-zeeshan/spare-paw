"""Tests for _md_to_html: standard Markdown → Telegram-supported HTML."""

from spare_paw.bot.handler import _md_to_html


class TestMdToHtml:
    def test_bold(self):
        assert _md_to_html("**hello**") == "<b>hello</b>"

    def test_italic(self):
        assert _md_to_html("*hello*") == "<i>hello</i>"

    def test_inline_code(self):
        assert _md_to_html("`foo()`") == "<code>foo()</code>"

    def test_code_block(self):
        md = "```python\nprint('hi')\n```"
        expected = '<pre><code class="language-python">print(\'hi\')</code></pre>'
        assert _md_to_html(md) == expected

    def test_code_block_no_language(self):
        md = "```\nprint('hi')\n```"
        expected = "<pre><code>print('hi')</code></pre>"
        assert _md_to_html(md) == expected

    def test_link(self):
        assert _md_to_html("[click](https://example.com)") == '<a href="https://example.com">click</a>'

    def test_strikethrough(self):
        assert _md_to_html("~~deleted~~") == "<s>deleted</s>"

    def test_html_escaping(self):
        result = _md_to_html("<script>")
        assert result == "&lt;script&gt;"

    def test_code_block_preserves_content(self):
        md = "```\n**not bold** and *not italic*\n```"
        expected = "<pre><code>**not bold** and *not italic*</code></pre>"
        assert _md_to_html(md) == expected

    def test_plain_text_unchanged(self):
        assert _md_to_html("hello world") == "hello world"

    def test_mixed_formatting(self):
        md = "Use **bold** and `code` and [link](https://example.com) together"
        result = _md_to_html(md)
        assert "<b>bold</b>" in result
        assert "<code>code</code>" in result
        assert '<a href="https://example.com">link</a>' in result

    def test_heading_h3(self):
        assert _md_to_html("### Time Complexity") == "<b>Time Complexity</b>"

    def test_heading_h2(self):
        assert _md_to_html("## Section Title") == "<b>Section Title</b>"

    def test_heading_h1(self):
        assert _md_to_html("# Main Title") == "<b>Main Title</b>"

    def test_bullet_list_not_italic(self):
        """Bullet items starting with * should NOT become italic."""
        md = "*   Item one\n*   Item two"
        result = _md_to_html(md)
        assert "<i>" not in result
        assert "Item one" in result
        assert "Item two" in result

    def test_latex_dollar_signs(self):
        """$O(n)$ math notation should pass through without breaking."""
        md = "Time complexity is $O(n)$ and space is $O(1)$."
        result = _md_to_html(md)
        assert "O(n)" in result
        assert "O(1)" in result

    def test_heading_with_inline_bold(self):
        md = "### Time Complexity: **O(n)**"
        result = _md_to_html(md)
        assert "<b>" in result
        assert "O(n)" in result

    def test_simple_table(self):
        """Markdown tables should be converted to a readable monospace format."""
        md = "| Model | Price |\n|-------|-------|\n| GPT-4 | $10 |\n| Claude | $8 |"
        result = _md_to_html(md)
        assert "GPT-4" in result
        assert "Claude" in result
        # Should use <pre> for alignment
        assert "<pre>" in result

    def test_table_preserves_content(self):
        """Table cell content should not be lost."""
        md = "| Name | Value |\n|------|-------|\n| foo | 42 |"
        result = _md_to_html(md)
        assert "foo" in result
        assert "42" in result

    def test_code_block_escapes_html_chars(self):
        """< and > inside code blocks must be escaped for Telegram HTML."""
        md = "```python\nif n <= 0:\n    return n > 1\n```"
        result = _md_to_html(md)
        assert "&lt;" in result
        assert "&gt;" in result
        assert "<=" not in result.replace("&lt;=", "")
