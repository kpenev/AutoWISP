from sqlalchemy import create_engine
from base import DataModelBase
from image_type import ImageType
from image import Image
from condition_expressions import ConditionExpressions
from conditions import Conditions
from step_type import StepType
from image_proc_progress import ImageProcProgress
from observing_session import ObservingSession
from image_conditions import ImageConditions
from configuration import Configuration

from sqlalchemy.orm import sessionmaker

engine = create_engine("sqlite:///automateDb.db", echo=True)
DataModelBase.metadata.create_all(bind=engine)

Session = sessionmaker(bind=engine)
session = Session()