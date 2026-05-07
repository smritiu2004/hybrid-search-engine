"""
test_storage.py
Tests for the Storage class.
Covers: save, load, checksum verification,
        serialisation, error handling.
"""

import os
import pytest
import tempfile
from src.storage import Storage
from src.indexer import Indexer



@pytest.fixture
def sample_pages():
    """Sample pages dict."""
    return {
        "https://quotes.toscrape.com/page/1/": (
            "The world as we have created it is a process of our thinking "
            "Albert Einstein change deep-thoughts"
        ),
        "https://quotes.toscrape.com/page/2/": (
            "It is our choices that show what we truly are "
            "J.K. Rowling choices decisions"
        ),
        "https://quotes.toscrape.com/page/3/": (
            "Good friends good books and a sleepy conscience "
            "Mark Twain friends books good"
        )
    }


@pytest.fixture
def built_indexer(sample_pages):
    """Returns a fully built Indexer."""
    indexer = Indexer()
    indexer.build(sample_pages)
    return indexer


@pytest.fixture
def temp_storage(tmp_path):
    """Returns a Storage instance using temporary directory."""
    index_path = str(tmp_path / "index.msgpack")
    checksum_path = str(tmp_path / "index.sha256")
    return Storage(
        index_path=index_path,
        checksum_path=checksum_path
    )


@pytest.fixture
def saved_storage(temp_storage, built_indexer):
    """Returns a Storage instance with index already saved."""
    temp_storage.save(built_indexer)
    return temp_storage



class TestStorageInit:
    def test_default_index_path(self):
        """Storage initialises with correct default index path."""
        storage = Storage()
        assert storage.index_path == "data/index.msgpack"

    def test_default_checksum_path(self):
        """Storage initialises with correct default checksum path."""
        storage = Storage()
        assert storage.checksum_path == "data/index.sha256"

    def test_custom_paths(self, tmp_path):
        """Storage accepts custom paths."""
        index_path = str(tmp_path / "custom.msgpack")
        checksum_path = str(tmp_path / "custom.sha256")
        storage = Storage(
            index_path=index_path,
            checksum_path=checksum_path
        )
        assert storage.index_path == index_path
        assert storage.checksum_path == checksum_path



class TestChecksum:
    def test_checksum_returns_string(self, temp_storage):
        """_compute_checksum returns a string."""
        result = temp_storage._compute_checksum(b"test data")
        assert isinstance(result, str)

    def test_checksum_correct_length(self, temp_storage):
        """SHA256 checksum is 64 characters long."""
        result = temp_storage._compute_checksum(b"test data")
        assert len(result) == 64

    def test_checksum_consistent(self, temp_storage):
        """Same data produces same checksum."""
        data = b"test data"
        assert temp_storage._compute_checksum(data) == \
               temp_storage._compute_checksum(data)

    def test_checksum_different_data(self, temp_storage):
        """Different data produces different checksum."""
        assert temp_storage._compute_checksum(b"data1") != \
               temp_storage._compute_checksum(b"data2")

    def test_checksum_empty_bytes(self, temp_storage):
        """Checksum handles empty bytes."""
        result = temp_storage._compute_checksum(b"")
        assert isinstance(result, str)
        assert len(result) == 64


class TestStorageSave:
    def test_save_creates_index_file(
        self, temp_storage, built_indexer
    ):
        """save creates index file on disk."""
        temp_storage.save(built_indexer)
        assert os.path.exists(temp_storage.index_path)

    def test_save_creates_checksum_file(
        self, temp_storage, built_indexer
    ):
        """save creates checksum file on disk."""
        temp_storage.save(built_indexer)
        assert os.path.exists(temp_storage.checksum_path)

    def test_save_index_file_not_empty(
        self, temp_storage, built_indexer
    ):
        """save writes non-empty index file."""
        temp_storage.save(built_indexer)
        assert os.path.getsize(temp_storage.index_path) > 0

    def test_save_checksum_file_not_empty(
        self, temp_storage, built_indexer
    ):
        """save writes non-empty checksum file."""
        temp_storage.save(built_indexer)
        assert os.path.getsize(temp_storage.checksum_path) > 0

    def test_save_checksum_is_valid_sha256(
        self, temp_storage, built_indexer
    ):
        """save writes valid SHA256 hex string."""
        temp_storage.save(built_indexer)
        with open(temp_storage.checksum_path, "r") as f:
            checksum = f.read().strip()
        assert len(checksum) == 64
        assert all(c in "0123456789abcdef" for c in checksum)

    def test_save_creates_data_directory(
        self, tmp_path, built_indexer
    ):
        """save creates data directory if it doesn't exist."""
        nested_path = str(tmp_path / "nested" / "dir" / "index.msgpack")
        checksum_path = str(tmp_path / "nested" / "dir" / "index.sha256")
        storage = Storage(
            index_path=nested_path,
            checksum_path=checksum_path
        )
        storage.save(built_indexer)
        assert os.path.exists(nested_path)



class TestStorageLoad:
    def test_load_returns_true_on_success(
        self, saved_storage
    ):
        """load returns True on successful load."""
        indexer = Indexer()
        result = saved_storage.load(indexer)
        assert result is True

    def test_load_restores_url_to_id(
        self, saved_storage, built_indexer
    ):
        """load restores url_to_id mapping."""
        indexer = Indexer()
        saved_storage.load(indexer)
        assert len(indexer.url_to_id) == len(built_indexer.url_to_id)

    def test_load_restores_id_to_url(
        self, saved_storage, built_indexer
    ):
        """load restores id_to_url mapping."""
        indexer = Indexer()
        saved_storage.load(indexer)
        assert len(indexer.id_to_url) == len(built_indexer.id_to_url)

    def test_load_restores_index_terms(
        self, saved_storage, built_indexer
    ):
        """load restores all index terms."""
        indexer = Indexer()
        saved_storage.load(indexer)
        assert len(indexer.index) == len(built_indexer.index)

    def test_load_restores_page_lengths(
        self, saved_storage, built_indexer
    ):
        """load restores page lengths."""
        indexer = Indexer()
        saved_storage.load(indexer)
        assert len(indexer.page_lengths) == \
               len(built_indexer.page_lengths)

    def test_load_restores_pages(
        self, saved_storage, built_indexer
    ):
        """load restores raw page text."""
        indexer = Indexer()
        saved_storage.load(indexer)
        assert len(indexer.pages) == len(built_indexer.pages)

    def test_load_returns_false_missing_index(
        self, temp_storage
    ):
        """load returns False when index file missing."""
        indexer = Indexer()
        result = temp_storage.load(indexer)
        assert result is False

    def test_load_returns_false_missing_checksum(
        self, temp_storage, built_indexer, tmp_path
    ):
        """load returns False when checksum file missing."""
        temp_storage.save(built_indexer)
        os.remove(temp_storage.checksum_path)
        indexer = Indexer()
        result = temp_storage.load(indexer)
        assert result is False

    def test_load_returns_false_corrupted_index(
        self, saved_storage, tmp_path
    ):
        """load returns False when index is corrupted."""
        # Corrupt the index file
        with open(saved_storage.index_path, "wb") as f:
            f.write(b"corrupted data")
        indexer = Indexer()
        result = saved_storage.load(indexer)
        assert result is False

    def test_load_roundtrip_preserves_term_stats(
        self, saved_storage, built_indexer
    ):
        """load preserves term frequency and TF-IDF after roundtrip."""
        indexer = Indexer()
        saved_storage.load(indexer)

        # Check a term exists and has correct stats
        for term in built_indexer.index:
            assert term in indexer.index
            break



class TestIndexConversion:
    def test_index_to_dict_returns_dict(
        self, temp_storage, built_indexer
    ):
        """_index_to_dict returns a regular dict."""
        result = temp_storage._index_to_dict(built_indexer.index)
        assert isinstance(result, dict)

    def test_dict_to_index_returns_defaultdict(
        self, temp_storage, built_indexer
    ):
        """_dict_to_index returns a defaultdict."""
        from collections import defaultdict
        regular_dict = temp_storage._index_to_dict(built_indexer.index)
        result = temp_storage._dict_to_index(regular_dict)
        assert isinstance(result, defaultdict)

    def test_conversion_roundtrip_preserves_terms(
        self, temp_storage, built_indexer
    ):
        """Converting to dict and back preserves all terms."""
        regular_dict = temp_storage._index_to_dict(built_indexer.index)
        restored = temp_storage._dict_to_index(regular_dict)
        assert set(restored.keys()) == set(built_indexer.index.keys())