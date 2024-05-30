import sqlalchemy
from sqlalchemy import create_engine
engine = create_engine('mysql://scott:tiger@localhost/foo', echo=True)
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

class ImagingDevices(Base):
	_tablename_='Imgdevices'
	id=Column('id',Integer,primary_key=True)
	x_res=Column('x_res',Integer)
	y_res=Column('y_res',Integer)
	x_pixel_size=Column('x_pixel_size',Float) //add channells//
		y_pixel_size=Column('y_pixel_size',Float)
	bit_depth=Column('bit_depth',Integer)//changes//
    Saturation=Column('Saturation',BigInteger)
    Charge_Conserving=Column('CC',Boolean)
    Serial_Number=Column('Serial_Number',Varchar)
    Firmware_version=Column('Firmware_version',Varchar)
    read_noise=Column('read_noise',Float)
    Shutter_open_time=Column('SOT',Integer)
    Shutter_close_time=Column('SCT',Integer)
    Temperature_control=Column('TC',Boolean)
    //Overscan area separate table//
    time_stamp=Column('TS',Integer)
 class Telescope(Base):
 	__tablename_='Telescope'
    id=Column('id',Integer,primary_key=True)
    focal_length=Column('focal_length',FLoat)
    f_ratio=Column('f-ratio',Float)
    time_stamp=Column('TS',Integer)
 class Session(Base):
    _tablename_='Session'
    id=Column('id',Integer,primary_key=True)
    Start_time=Column('Start_time',Float)
    Time_duration=Column('Time_duration',FLoat)
    time_stamp=Column('TS',Integer)
    image_number=Column('number of images',Integer)
class Location(Base):
	_tablename_='Location'
	id=Column('id',Integer,primary_key=True)
	Latitude=Column('Latitude',FLoat)
	Longtitude=Column('Longitude',FLoat)
	time_stamp=Column("TS",integer)
class Mount(Base):
    _tablename_='Mount'
    id=Column('id',Integer,primary_key=True)
    Mount_type=('mount_type',Varchar)
    Automation=('automated',Boolean)





