from datetime import datetime

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

engine = create_engine("sqlite:///testDb.db", echo=True)
DataModelBase.metadata.create_all(bind=engine)

Session = sessionmaker(bind=engine)
session = Session()

#Image Type
#id, type_name, description, timestamp
img_typ1 = ImageType(1, "type_name 1", "This is the first img", datetime.utcnow())
img_typ2 = ImageType(2, "type_name 2", "This is the second img", datetime.utcnow())
img_typ3 = ImageType(3, "type_name 3", "This is the third img", datetime.utcnow())
#Condition_Expressions
#id, expression, notes, timestamp
cond_expr1 = ConditionExpressions(1, "expression 1", "here is a note for the expression", datetime.utcnow())
cond_expr2 = ConditionExpressions(2, "expression 2", "here is another note for the expression", datetime.utcnow())
cond_expr3 = ConditionExpressions(3, "expression 3", "here is note for the expression again", datetime.utcnow())
#Observing_Session
#id, observer_id, camera_id, telescope_id, mount_id, obseratory_id, target_id, start_time, end_time, notes, timestamp
obssess1 = ObservingSession(1,0,0,0,0,0,0,datetime.utcnow(),datetime.utcnow(), "note for obs sess 1", datetime.utcnow())
obssess2 = ObservingSession(2,0,0,0,0,0,0,datetime.utcnow(),datetime.utcnow(), "note for obs sess 2", datetime.utcnow())
obssess3 = ObservingSession(3,0,0,0,0,0,0,datetime.utcnow(),datetime.utcnow(), "note for obs sess 3", datetime.utcnow())
#Step_Type
#id, notes, timestamp
steptype1 = StepType(1, "note for step_type", datetime.utcnow())
steptype2 = StepType(2, "a different note for step_type", datetime.utcnow())
#Conditions
#id, expression_id, notes, timestamp
cond1 = Conditions(1, cond_expr2.id, "a note for the condition", datetime.utcnow())
cond2 = Conditions(2, cond_expr1.id, "another note for the condition", datetime.utcnow())
#Image
#id, image_type_id, observing_session_id, notes, timestamp
img1 = Image(1, img_typ1.id, obssess1.id,"This is a note", datetime.utcnow())
img2 = Image(2, img_typ1.id, obssess2.id,"This is a note again", datetime.utcnow())
img3 = Image(3, img_typ3.id, obssess3.id,"This is another note", datetime.utcnow())
#Image_Conditions
#id, image_id, condition_id, timestamp
imgcond1 = ImageConditions(1, img1.id, cond1.id, datetime.utcnow())
imgcond2 = ImageConditions(2, img3.id, cond1.id, datetime.utcnow())
imgcond3 = ImageConditions(3, img2.id, cond2.id, datetime.utcnow())
#Configuration
#id, version, condition_id, parameter, value, notes, timestamp
config1 = Configuration(1, 0, cond1.id, "param config1", "value config1", "note config1", datetime.utcnow())
config2 = Configuration(2, 0, cond2.id, "param config2", "value config2", "note config2", datetime.utcnow())
config3 = Configuration(3, 0, cond2.id, "param config3", "value config3", "note config3", datetime.utcnow())
#Image_Processing_Progress
#id, image_id, step_type_id, config_version, timestamp
imgprocprog1 = ImageProcProgress(1, img1.id, steptype2.id, config2.version, datetime.utcnow())
imgprocprog2 = ImageProcProgress(2, img3.id, steptype2.id, config1.version, datetime.utcnow())
imgprocprog3 = ImageProcProgress(3, img2.id, steptype1.id, config3.version, datetime.utcnow())

#Add stuff to db
session.add(img_typ1)
session.add(img_typ2)
session.add(img_typ3)
session.add(cond_expr1)
session.add(cond_expr2)
session.add(cond_expr3)
session.add(obssess1)
session.add(obssess2)
session.add(obssess3)
session.add(steptype1)
session.add(steptype2)
session.add(cond1)
session.add(cond2)
session.add(img1)
session.add(img2)
session.add(img3)
session.add(imgcond1)
session.add(imgcond2)
session.add(imgcond3)
session.add(config1)
session.add(config2)
session.add(config3)
session.add(imgprocprog1)
session.add(imgprocprog2)
session.add(imgprocprog3)
session.commit()

#
# #**WORKS** 06/09/23
# #ImageType --> Image
# #id, type_name, description, timestamp
# img_typ1 = ImageType(1, 23, "This is the first img", datetime.utcnow())
# img_typ2 = ImageType(2, 59, "This is the second img", datetime.utcnow())
# img_typ3 = ImageType(3, 82, "This is the third img", datetime.utcnow())
#
# #id, image_type_id, observing_session_id, notes, timestamp
# img1 = Image(1, img_typ1.id, 574,"This is a note", datetime.utcnow())
# img2 = Image(2, img_typ1.id, 52,"This is a note again", datetime.utcnow())
# img3 = Image(3, img_typ3.id, 23,"This is another note", datetime.utcnow())
#
# session.add(img1)
# session.add(img2)
# session.add(img3)
# session.add(img_typ1)
# session.add(img_typ2)
# session.add(img_typ3)
# session.commit()
#
# #**WORKS** 06/09/23
# #ConditionExpression --> Condition
# #id, expression, notes, timestamp
# cond_expr1 = ConditionExpressions(1, "expression 1", "here is a note for the expression", datetime.utcnow())
# cond_expr2 = ConditionExpressions(2, "expression 2", "here is another note for the expression", datetime.utcnow())
# cond_expr3 = ConditionExpressions(3, "expression 3", "here is note for the expression again", datetime.utcnow())
#
# #id, expression_id, notes, timestamp
# cond1 = Conditions(1, cond_expr2.id, "a note for the condition", datetime.utcnow())
# cond2 = Conditions(2, cond_expr1.id, "another note for the condition", datetime.utcnow())
#
# session.add(cond_expr1)
# session.add(cond_expr2)
# session.add(cond_expr3)
# session.add(cond1)
# session.add(cond2)
# session.commit()
#
# #**WORKS** 06/09/2023
# #StepType --> ImageProcessingProgress
# #id, notes, timestamp
# steptype1 = StepType(1, "note for step_type", datetime.utcnow())
# steptype2 = StepType(2, "a different note for step_type", datetime.utcnow())
# # #id, image_id, step_type_id, config_version, timestamp
# imgprocprog1 = ImageProcProgress(1, 1, steptype1.id, 1, datetime.utcnow())
# imgprocprog2 = ImageProcProgress(2, -1, steptype1.id, -1, datetime.utcnow())
# imgprocprog3 = ImageProcProgress(3, -1, steptype2.id, -1, datetime.utcnow())
#
# session.add(steptype1)
# session.add(steptype2)
# session.add(imgprocprog1)
# session.add(imgprocprog2)
# session.add(imgprocprog3)
# session.commit()
#
#**WORKS** 06/09/2023
# #ObservingSession --> Image
# #id, observer_id, camera_id, telescope_id, mount_id, obseratory_id, target_id, start_time, end_time, notes, timestamp
# obssess1 = ObservingSession(1,0,0,0,0,0,0,datetime.utcnow(),datetime.utcnow(), "this is a note for obs sess", datetime.utcnow())
# obssess2 = ObservingSession(2,0,0,0,0,0,0,datetime.utcnow(),datetime.utcnow(), "this is another note for obs sess", datetime.utcnow())
# obssess3 = ObservingSession(3,0,0,0,0,0,0,datetime.utcnow(),datetime.utcnow(), "this is a note again for obs sess", datetime.utcnow())
# #id, image_type_id, observing_session_id, notes, timestamp
# img1 = Image(1, img_typ1.id, obssess1.id,"This is a note", datetime.utcnow())
# img2 = Image(2, img_typ1.id, obssess2.id,"This is a note again", datetime.utcnow())
# img3 = Image(3, img_typ3.id, obssess3.id,"This is another note", datetime.utcnow())
#
# session.add(obssess1)
# session.add(obssess2)
# session.add(obssess3)
# session.add(img1)
# session.add(img2)
# session.add(img3)
# session.commit()
#
# # **WORKS** 06/11-23
# # Conditions -> Image Conditions
# #id, expression_id, notes, timestamp
# cond1 = Conditions(1, cond_expr2.id, "a note for the condition", datetime.utcnow())
# cond2 = Conditions(2, cond_expr1.id, "another note for the condition", datetime.utcnow())
#
# #id, condition_id, timestamp
# imgcond1 = ImageConditions(1, cond1.id, datetime.utcnow())
# imgcond2 = ImageConditions(2, cond1.id, datetime.utcnow())
# imgcond3 = ImageConditions(3, cond2.id, datetime.utcnow())
#
# session.add(cond1)
# session.add(cond2)
# session.add(imgcond1)
# session.add(imgcond2)
# session.add(imgcond3)
# session.commit()
#
# # **WORKS** 06/11/23
# # Image -> Image Conditions
# #id, iamge_id, condition_id, timestamp
# imgcond1 = ImageConditions(1, img3.id, cond1.id, datetime.utcnow())
# imgcond2 = ImageConditions(2, img1.id, cond2.id, datetime.utcnow())
# imgcond3 = ImageConditions(3, img2.id, cond2.id, datetime.utcnow())
#
# session.add(imgcond1)
# session.add(imgcond2)
# session.add(imgcond3)
# session.commit()
# #  **WORKS** 06/11/23
# # Image -> ImageProcessingProgress
# #id, image_type_id, observing_session_id, notes, timestamp
# img1 = Image(1, 0, 574,"This is a note", datetime.utcnow())
# img2 = Image(2, 0, 52,"This is a note again", datetime.utcnow())
# img3 = Image(3, 0, 23,"This is another note", datetime.utcnow())
#
# # #id, image_id, step_type_id, config_version, timestamp
# imgprocprog1 = ImageProcProgress(1, img1.id, 0, -1, datetime.utcnow())
# imgprocprog2 = ImageProcProgress(2, img3.id, 0, -1, datetime.utcnow())
# imgprocprog3 = ImageProcProgress(3, img2.id, 0, -1, datetime.utcnow())
#
# session.add(img1)
# session.add(img2)
# session.add(img3)
# session.add(imgprocprog1)
# session.add(imgprocprog2)
# session.add(imgprocprog3)
# session.commit()
#
# session.add(img1)
# session.add(img2)
# session.add(img3)
# # **WORKS** 06/11/23
# #Conditions -> Configuration -> ImgProcProg
# #id, expression_id, notes, timestamp
# cond1 = Conditions(1, 0, "a note for the condition", datetime.utcnow())
# cond2 = Conditions(2, 0, "another note for the condition", datetime.utcnow())
# #id, version, condition_id, parameter, value, notes, timestamp
# config1 = Configuration(1, 0, cond1.id, "this is a param for config1", "this a value for config1", "this a note for config1", datetime.utcnow())
# config2 = Configuration(2, 0, cond2.id, "this is a param for config2", "this a value for config2", "this a note for config2", datetime.utcnow())
# config3 = Configuration(3, 0, cond1.id, "this is a param for config3", "this a value for config3", "this a note for config3", datetime.utcnow())
# # #id, image_id, step_type_id, config_version, timestamp
# imgprocprog1 = ImageProcProgress(1, 0, 0, config2.version, datetime.utcnow())
# imgprocprog2 = ImageProcProgress(2, 0, 0, config1.version, datetime.utcnow())
# imgprocprog3 = ImageProcProgress(3, 0, 0, config3.version, datetime.utcnow())
#
# session.add(cond1)
# session.add(cond2)
# session.add(config1)
# session.add(config2)
# session.add(config3)
# session.add(imgprocprog1)
# session.add(imgprocprog2)
# session.add(imgprocprog3)
# session.commit()

# results = session.query(Image, ImageType).filter(Image.image_type_id == ImageType.id).filter(Image.image_type_id == 1).all()
# for r in results:
#     print(r)
