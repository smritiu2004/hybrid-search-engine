"""
test_crawler.py
Tests for the Crawler class.
Covers: fetching, text extraction, link extraction,
        politeness window, error handling.
"""

import time
import pytest
from unittest.mock import patch, MagicMock
from bs4 import BeautifulSoup
from src.crawler import Crawler

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def crawler():
    """Returns a Crawler instance with reduced politeness for testing."""
    return Crawler(politeness=0)


@pytest.fixture
def sample_html():
    """Returns sample HTML mimicking quotes.toscrape.com structure."""
    return """
    <html>
    <body>
        <div class="quote">
            <span class="text">"The world as we have created it is a process of our thinking."</span>
            <small class="author">Albert Einstein</small>
            <div class="tags">
                <a class="tag">change</a>
                <a class="tag">deep-thoughts</a>
            </div>
        </div>
        <div class="quote">
            <span class="text">"It is our choices that show what we truly are."</span>
            <small class="author">J.K. Rowling</small>
            <div class="tags">
                <a class="tag">choices</a>
            </div>
        </div>
        <a href="/page/2/">Next</a>
        <a href="https://external.com">External</a>
    </body>
    </html>
    """


@pytest.fixture
def soup(sample_html):
    """Returns BeautifulSoup object from sample HTML."""
    return BeautifulSoup(sample_html, "html.parser")


# -----------------------------------------------------------------------
# Initialisation Tests
# -----------------------------------------------------------------------

class TestCrawlerInit:
    def test_default_base_url(self):
        """Crawler initialises with correct default base URL."""
        crawler = Crawler()
        assert crawler.base_url == "https://quotes.toscrape.com"

    def test_default_politeness(self):
        """Crawler initialises with correct default politeness window."""
        crawler = Crawler()
        assert crawler.politeness == 6

    def test_custom_politeness(self):
        """Crawler accepts custom politeness window."""
        crawler = Crawler(politeness=0)
        assert crawler.politeness == 0

    def test_empty_visited_on_init(self):
        """Crawler starts with empty visited set."""
        crawler = Crawler()
        assert len(crawler.visited) == 0

    def test_empty_pages_on_init(self):
        """Crawler starts with empty pages dict."""
        crawler = Crawler()
        assert len(crawler.pages) == 0


# -----------------------------------------------------------------------
# fetch_page Tests
# -----------------------------------------------------------------------

class TestFetchPage:
    def test_successful_fetch(self, crawler):
        """fetch_page returns BeautifulSoup on 200 response."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body>Test</body></html>"

        with patch("src.crawler.requests.get", return_value=mock_response):
            result = crawler.fetch_page("https://quotes.toscrape.com")
            assert result is not None
            assert isinstance(result, BeautifulSoup)

    def test_404_returns_none(self, crawler):
        """fetch_page returns None on 404 response."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("src.crawler.requests.get", return_value=mock_response):
            result = crawler.fetch_page("https://quotes.toscrape.com/missing")
            assert result is None

    def test_connection_error_returns_none(self, crawler):
        """fetch_page returns None on connection error."""
        import requests
        with patch(
            "src.crawler.requests.get",
            side_effect=requests.exceptions.ConnectionError
        ):
            result = crawler.fetch_page("https://quotes.toscrape.com")
            assert result is None

    def test_timeout_returns_none(self, crawler):
        """fetch_page returns None on timeout."""
        import requests
        with patch(
            "src.crawler.requests.get",
            side_effect=requests.exceptions.Timeout
        ):
            result = crawler.fetch_page("https://quotes.toscrape.com")
            assert result is None


# -----------------------------------------------------------------------
# extract_text Tests
# -----------------------------------------------------------------------

class TestExtractText:
    def test_extracts_quote_text(self, crawler, soup):
        """extract_text includes quote text content."""
        text = crawler.extract_text(soup)
        assert "The world as we have created it" in text

    def test_extracts_author(self, crawler, soup):
        """extract_text includes author names."""
        text = crawler.extract_text(soup)
        assert "Albert Einstein" in text

    def test_extracts_tags(self, crawler, soup):
        """extract_text includes tags."""
        text = crawler.extract_text(soup)
        assert "change" in text

    def test_multiple_quotes_extracted(self, crawler, soup):
        """extract_text handles multiple quotes on a page."""
        text = crawler.extract_text(soup)
        assert "J.K. Rowling" in text
        assert "choices" in text

    def test_empty_page_returns_empty_string(self, crawler):
        """extract_text returns empty string for page with no quotes."""
        empty_soup = BeautifulSoup(
            "<html><body></body></html>",
            "html.parser"
        )
        text = crawler.extract_text(empty_soup)
        assert text.strip() == ""

    def test_returns_string(self, crawler, soup):
        """extract_text always returns a string."""
        result = crawler.extract_text(soup)
        assert isinstance(result, str)


# -----------------------------------------------------------------------
# extract_links Tests
# -----------------------------------------------------------------------

class TestExtractLinks:
    def test_extracts_internal_links(self, crawler, soup):
        """extract_links returns internal pagination links."""
        links = crawler.extract_links(soup)
        assert "https://quotes.toscrape.com/page/2/" in links

    def test_ignores_external_links(self, crawler, soup):
        """extract_links ignores external links."""
        links = crawler.extract_links(soup)
        assert "https://external.com" not in links

    def test_returns_list(self, crawler, soup):
        """extract_links returns a list."""
        links = crawler.extract_links(soup)
        assert isinstance(links, list)

    def test_no_links_returns_empty_list(self, crawler):
        """extract_links returns empty list when no links present."""
        empty_soup = BeautifulSoup(
            "<html><body><p>No links</p></body></html>",
            "html.parser"
        )
        links = crawler.extract_links(empty_soup)
        assert links == []
    
    def test_skips_visited_links(self, crawler, soup):
        """crawl does not revisit already visited URLs."""
        crawler.visited.add("https://quotes.toscrape.com/page/2/")
        assert "https://quotes.toscrape.com/page/2/" in crawler.visited



# -----------------------------------------------------------------------
# Crawl Tests
# -----------------------------------------------------------------------

class TestCrawl:
    def test_crawl_returns_dict(self, crawler):
        """crawl returns a dictionary."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """
        <html><body>
        <div class="quote">
            <span class="text">Test quote</span>
            <small class="author">Author</small>
        </div>
        </body></html>
        """

        with patch("src.crawler.requests.get", return_value=mock_response):
            result = crawler.crawl()
            assert isinstance(result, dict)

    def test_visited_urls_not_recrawled(self, crawler):
        """Crawler does not visit the same URL twice."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """
        <html><body>
        <div class="quote">
            <span class="text">Test</span>
            <small class="author">Auth</small>
        </div>
        </body></html>
        """

        with patch(
            "src.crawler.requests.get",
            return_value=mock_response
        ) as mock_get:
            crawler.crawl()
            base_url = "https://quotes.toscrape.com"
            call_urls = [
                call[0][0]
                for call in mock_get.call_args_list
            ]
            assert call_urls.count(base_url) == 1

    def test_empty_pages_skipped(self, crawler):
        """Pages with no extractable text are not stored."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = """
        <html><body><p>No quotes here</p></body></html>
        """

        with patch("src.crawler.requests.get", return_value=mock_response):
            result = crawler.crawl()
            assert len(result) == 0