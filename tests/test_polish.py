def test_cost_accounting(tmp_path, monkeypatch):
    import clerk.store as store
    from clerk.config import Settings
    s = Settings(data_dir=str(tmp_path))
    monkeypatch.setattr("clerk.config.settings", s)
    for mod in ("clerk.store", "clerk.engine"):
        monkeypatch.setattr(f"{mod}.settings", s)
    store.init_db(s.db_path)
    qid = store.log_query("q?", "a", "ok", [], 100, 1_000_000, 500_000,
                          cost_usd=(1e6 * s.price_in_per_mtok + 5e5 * s.price_out_per_mtok) / 1e6)
    st = store.stats()
    assert st["total_cost_usd"] > 0
    assert st["status_counts"]["ok"] == 1


def test_rerank_passthrough_offline():
    """LLM down => rerank must return input order unchanged (graceful)."""
    import os
    os.environ["CLERK_LLM_BASE_URL"] = "http://127.0.0.1:1/v1"
    from clerk.config import Settings
    import clerk.config as cfg
    import clerk.retrieval as R
    orig = cfg.settings
    object.__setattr__(cfg.settings, "llm_base_url", "http://127.0.0.1:1/v1")
    rows = [{"text": f"chunk {i} about contracts and torts and procedure " * 3} for i in range(10)]
    try:
        out = R.llm_rerank("what about contracts?", rows, 6)
        assert [r["text"] for r in out] == [r["text"] for r in rows]
    finally:
        object.__setattr__(cfg.settings, "llm_base_url", orig.llm_base_url)


def test_db_migration_adds_cost_column(tmp_path, monkeypatch):
    """A legacy DB without cost_usd migrates cleanly."""
    import sqlite3
    import clerk.store as store
    from clerk.config import Settings
    s = Settings(data_dir=str(tmp_path))
    monkeypatch.setattr("clerk.config.settings", s)
    monkeypatch.setattr("clerk.store.settings", s)
    legacy = s.db_path
    con = sqlite3.connect(legacy)
    con.executescript(store._SCHEMA.replace("cost_usd    REAL DEFAULT 0,", ""))
    con.commit()
    con.close()
    store.init_db(legacy)
    assert "cost_usd" in {r[1] for r in sqlite3.connect(legacy).execute("PRAGMA table_info(queries)")}
