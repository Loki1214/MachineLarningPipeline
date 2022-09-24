from sqlalchemy import Column, Integer, String, DATETIME, BOOLEAN
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
class DigitImages_TABLE(Base):
	'''
	Column definition
	'''
	__abstract__=True
	__tablename__ = 'default'
	id      = Column('id',      Integer, primary_key=True)  # 主キー
	relpath = Column('relpath', String(200))
	label   = Column('label',   Integer)
	date    = Column('date',    DATETIME)
	is_used = Column('is_used', BOOLEAN)

class MNIST(DigitImages_TABLE):
	__tablename__ = 'MNIST'

class EMNIST(DigitImages_TABLE):
	__tablename__ = 'EMNIST'

class Uploaded(DigitImages_TABLE):
	__tablename__ = 'uploaded'