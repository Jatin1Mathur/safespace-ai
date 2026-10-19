import pytest


@pytest.fixture(autouse=True)
def isolated_memory(tmp_path, monkeypatch):
    """Keep tests from writing to the real session_memory.json."""
    import tools

    monkeypatch.setattr(tools, "MEMORY_FILE", str(tmp_path / "session_memory.json"))
    # Each test starts with the map servers "not known to be down"
    monkeypatch.setattr(tools, "_overpass_down_until", 0.0)
