import pytest


@pytest.fixture(autouse=True)
def isolated_memory(tmp_path, monkeypatch):
    """Keep tests from writing to the real session_memory.json."""
    import tools

    monkeypatch.setattr(tools, "MEMORY_FILE", str(tmp_path / "session_memory.json"))
