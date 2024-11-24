import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from third.io.query.table import metadata


@pytest.fixture(scope="function")
def session():
    engine = create_engine("sqlite:///:memory:")
    Session = sessionmaker(bind=engine)
    metadata.create_all(engine)
    session = Session()
    yield session
    session.close()
    metadata.drop_all(engine)
    engine.dispose()
