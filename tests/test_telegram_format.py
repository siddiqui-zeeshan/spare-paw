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

    def test_inline_code_escapes_html_chars(self):
        """< and > inside inline code must be escaped."""
        result = _md_to_html("`a < b && c > d`")
        assert "<code>" in result
        assert "&lt;" in result
        assert "&gt;" in result
        assert "&amp;" in result

    def test_multiple_code_blocks(self):
        """Multiple code blocks in one message should all be preserved."""
        md = "First:\n```python\nx = 1\n```\nSecond:\n```js\ny = 2\n```"
        result = _md_to_html(md)
        assert "x = 1" in result
        assert "y = 2" in result
        assert result.count("<pre>") == 2

    def test_empty_input(self):
        assert _md_to_html("") == ""

    def test_ampersand_in_text(self):
        """Ampersands in regular text must be escaped."""
        result = _md_to_html("AT&T and H&M")
        assert "AT&amp;T" in result
        assert "H&amp;M" in result

    def test_numbered_list(self):
        """Numbered lists should pass through without breaking."""
        md = "1. First item\n2. Second item\n3. Third item"
        result = _md_to_html(md)
        assert "1. First item" in result
        assert "3. Third item" in result

    def test_bold_italic_combined(self):
        """***bold italic*** should produce nested tags."""
        result = _md_to_html("***hello***")
        assert "<b>" in result or "<i>" in result
        assert "hello" in result

    def test_inline_code_not_formatted(self):
        """Bold/italic markers inside inline code should not be converted."""
        result = _md_to_html("`**not bold**`")
        assert "<b>" not in result
        assert "<code>" in result

    def test_real_world_llm_response(self):
        """A realistic multi-paragraph LLM response with mixed formatting."""
        md = (
            "## Summary\n\n"
            "Here's what I found:\n\n"
            "**Model A** costs `$0.50` per million tokens. "
            "See [pricing](https://example.com/pricing).\n\n"
            "```python\ndef calc(n):\n    return n * 0.50\n```\n\n"
            "| Model | Cost |\n|-------|------|\n| A | $0.50 |\n| B | $1.00 |\n\n"
            "~~Old pricing~~ is no longer valid."
        )
        result = _md_to_html(md)
        assert "<b>Summary</b>" in result
        assert "<b>Model A</b>" in result
        assert "<code>$0.50</code>" in result
        assert '<a href="https://example.com/pricing">' in result
        assert "<pre><code" in result
        assert "<s>Old pricing</s>" in result
        # Should not contain any raw < or > outside of tags
        import re
        # Strip all valid HTML tags, check nothing raw remains
        stripped = re.sub(r"<[a-z/][^>]*>", "", result)
        assert "<" not in stripped, f"Unescaped < found in: {stripped}"
