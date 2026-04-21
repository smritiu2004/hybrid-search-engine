"""
FileName: crawler.py
Crawls all pages of website https://quotes.toscrape.com/
Ensures a politeness window of 6 seconds between requests.
Extracts quotes, authors, and tags from each page.
"""

import time
import logging
import requests
from bs4 import BeautifulSoup

# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Creates a logger for this particular module created. 
# Done to ensure that logs from this module are easily identifiable 
# and can be configured separately if needed.
logger = logging.getLogger(__name__)

BASE_URL = "https://quotes.toscrape.com"
POLITENESS_WINDOW = 6  



class Crawler:
    def __init__(self, base_url: str = BASE_URL, politeness: int = POLITENESS_WINDOW):
        self.base_url = base_url
        self.politeness = politeness
        self.visited = set()
        self.pages = {} 

    def fetch_page(self, url: str) -> BeautifulSoup | None:
        """
        Fetches a single page and returns a BeautifulSoup object.
        Returns None if the request fails.
        """
        try:
            logger.info(f"Fetching: {url}")
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                return BeautifulSoup(response.text, "html.parser")
            else:
                logger.warning(f"Non-200 status {response.status_code} for {url}")
                return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def extract_text(self, soup: BeautifulSoup) -> str:
        """
        Extracts meaningful text from a page.
        Targets the quotes, authors, and tags specifically from each quote.
        """
        texts = []

        # To extract quotes
        for quote in soup.select(".quote"):
            text = quote.select_one(".text")
            author = quote.select_one(".author")
            tags = quote.select(".tags .tag")

            if text:
                texts.append(text.get_text(strip=True))
            if author:
                texts.append(author.get_text(strip=True))
            for tag in tags:
                texts.append(tag.get_text(strip=True))

        return " ".join(texts)

    def extract_links(self, soup: BeautifulSoup) -> list[str]:
        """
        Extracts all internal links from a page.
        """
        links = []
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            
            # Ensures only internal links are followed
            if href.startswith("/") and href not in self.visited:
                full_url = self.base_url + href
                links.append(full_url)
        return links

    def crawl(self) -> dict:
        """
        Crawls all pages starting from base_url.
        Returns a dict of {url: text_content}.
        """
        queue = [self.base_url]

        while queue:
            url = queue.pop(0)

            # Skipping already visited URLs to avoid cycles
            if url in self.visited:
                continue

            self.visited.add(url)
            soup = self.fetch_page(url)

            if soup is None:
                continue

            # Extracting and storing text
            text = self.extract_text(soup)
            if text.strip():
                self.pages[url] = text
                logger.info(f"Stored content for: {url}")

            # Extract new links
            new_links = self.extract_links(soup)
            for link in new_links:
                if link not in self.visited:
                    queue.append(link)

            # The 6 second politeness window
            logger.info(f"Waiting {self.politeness} seconds...")
            time.sleep(self.politeness)

        logger.info(f"Crawl complete. {len(self.pages)} pages indexed.")
        return self.pages
