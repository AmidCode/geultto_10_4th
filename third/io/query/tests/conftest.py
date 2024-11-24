import pytest
from sqlalchemy import (
    create_engine,
    text,
)
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


def create_database(default_url: str, db_name: str):
    """테스트 데이터베이스 생성"""
    engine = create_engine(default_url)
    conn = engine.connect()
    # auto-commit mode
    conn = conn.execution_options(isolation_level="AUTOCOMMIT")

    try:
        # 기존 연결 종료 시도
        conn.execute(
            text(
                f"""
            SELECT pg_terminate_backend(pid) 
            FROM pg_stat_activity 
            WHERE datname = '{db_name}'
        """
            )
        )
        # DB가 존재하면 삭제
        conn.execute(text(f"DROP DATABASE IF EXISTS {db_name}"))
        # 새로운 DB 생성
        conn.execute(text(f"CREATE DATABASE {db_name}"))
    finally:
        conn.close()
        engine.dispose()


def drop_database(default_url: str, db_name: str):
    """테스트 데이터베이스 삭제"""
    engine = create_engine(default_url)
    conn = engine.connect()
    conn = conn.execution_options(isolation_level="AUTOCOMMIT")
    try:
        conn.execute(
            text(
                f"""
            SELECT pg_terminate_backend(pid) 
            FROM pg_stat_activity 
            WHERE datname = '{db_name}'
        """
            )
        )
        conn.execute(text(f"DROP DATABASE IF EXISTS {db_name}"))
    finally:
        conn.close()
        engine.dispose()


@pytest.fixture(scope="session")
def database():
    """테스트 데이터베이스 생성/삭제 관리"""
    default_url = "postgresql://postgres:example@localhost:15432/postgres"
    test_db_name = "geultto"

    create_database(default_url, test_db_name)
    yield
    drop_database(default_url, test_db_name)


@pytest.fixture(scope="function")
def session(database):
    """테스트용 세션 생성"""
    db_url = "postgresql://postgres:example@localhost:15432/geultto"
    engine = create_engine(db_url, echo=True)

    # 테이블 생성
    metadata.create_all(engine)

    # 세션 생성
    Session = sessionmaker(bind=engine)
    session = Session()

    yield session

    # 클린업
    session.close()
    metadata.drop_all(engine)
    engine.dispose()
