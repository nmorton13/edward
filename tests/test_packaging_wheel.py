"""Tests verifying packaged data assets and distribution wheel completeness."""

import importlib.resources


def test_packaged_migrations_exist():
    """Verify migrations directory and SQL files are accessible within package."""
    package_files = importlib.resources.files("edward")
    migrations_dir = package_files / "migrations"
    assert migrations_dir.is_dir()

    migration_files = [f.name for f in migrations_dir.iterdir() if f.name.endswith(".sql")]
    assert len(migration_files) >= 1
    assert "001_initial_schema.sql" in migration_files


def test_packaged_registries_exist():
    """Verify classification registries are accessible within package."""
    package_files = importlib.resources.files("edward")
    registries_dir = package_files / "registries"
    assert registries_dir.is_dir()

    expected_registries = {
        "topics-v1.json",
        "forms-v1.json",
        "signals-v1.json",
        "jev-questions-v1.json",
        "thresholds-v1.json",
    }
    present = {f.name for f in registries_dir.iterdir() if f.name.endswith(".json")}
    missing = expected_registries - present
    assert not missing, f"Missing packaged registries: {missing}"
