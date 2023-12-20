"""Connect to the database and provide a session scope for queries."""

import logging
from functools import wraps
from traceback import format_exc
from time import sleep

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from sqlalchemy.exc import OperationalError

db_engine = create_engine(
    #'mysql+pymysql://superphotuser:pipeline@kartof.utdallas.edu/'
    #'SuperPhotPipeline',
#    'mysql+pymysql://kpenev:shakakaa@kartof.utdallas.edu/sandbox',
    'mysql+pymysql://superphot:kartof@kartof.utdallas.edu/SuperPhot',
    echo=True,
    poolclass=NullPool,
    pool_pre_ping=True,
    pool_recycle=3600
)

#pylint false positive - Session is actually a class name.
#pylint: disable=invalid-name
Session = sessionmaker(db_engine)
#pylint: enable=invalid-name

def retry_on_db_fail(func, timeout=10, num_retries=10):
    """Retry a function if it fails due to a database error."""

    @wraps(func)
    def retry_wrapper(*args, **kwargs):
        """Wrapper function."""

        for i in range(num_retries):
            try:
                return func(*args, **kwargs)
            except OperationalError:
                logging.getLogger(__name__).error(
                    'Database error in %s.%s: %s\n%s',
                    func.__module__,
                    func.__name__,
                    format_exc(),
                    (
                        'Declaring hopeless!' if i == num_retries - 1
                        else f'Retrying in {timeout}s {i:d}/{num_retries:d}...'
                    )
                )
                if i == num_retries - 1:
                    raise
                sleep(timeout)
        assert False

    return retry_wrapper
