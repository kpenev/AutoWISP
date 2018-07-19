"""Connect to the database and provide a session scope for queries."""

from contextlib import contextmanager
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

#pylint false positive - Session is actually a class name.
#pylint: disable=invalid-name
Session = sessionmaker()
#pylint: enable=invalid-name

db_engine = create_engine(
    'mysql+pymysql://kpenev:shakakaa@localhost/PHYS2325SPRING2018',
    echo=True
)

Session.configure(bind=db_engine)

@contextmanager
def db_session_scope():
    """Provide a transactional scope around a series of operations."""
    session = Session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()
