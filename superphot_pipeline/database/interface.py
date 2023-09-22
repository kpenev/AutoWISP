"""Connect to the database and provide a session scope for queries."""

from contextlib import contextmanager
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

#pylint false positive - Session is actually a class name.
#pylint: disable=invalid-name
Session = sessionmaker()
#pylint: enable=invalid-name

db_engine = create_engine(
    #'mysql+pymysql://superphotuser:pipeline@kartof.utdallas.edu/'
    #'SuperPhotPipeline',
    'mysql+pymysql://kpenev:shakakaa@kartof.utdallas.edu/sandbox',
#    'mysql+pymysql://superphot:kartof@kartof.utdallas.edu/SuperPhot',
    echo=True,
    poolclass=NullPool,
    pool_pre_ping=True
)

Session.configure(bind=db_engine)

#False positive
#pylint: disable=no-member
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
#pylint: enable=no-member
