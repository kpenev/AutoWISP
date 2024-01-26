"""Connect to the database and provide a session scope for queries."""

from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

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
