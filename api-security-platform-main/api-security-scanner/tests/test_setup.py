def test_settings_import():
    from config.settings import FLASK_PORT
    assert FLASK_PORT == 5000

def test_database_init():
    from database.db import Base, engine
    assert Base is not None
    assert engine is not None
