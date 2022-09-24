import os
# NOT thread safe
class DBManager(object):
	dialect  = 'mysql'
	username = os.getenv('MYSQL_USER')
	password = os.getenv('MYSQL_PASSWORD')
	database = os.getenv('MYSQL_DATABASE')
	host     = os.getenv('DATABASE_HOST')
	port     = 3306

	_unique_instance = None
	def __new__(cls, *args, **kargs):
		if cls._unique_instance is None:
			cls._unique_instance = super(DBManager, cls).__new__(cls)
		return cls._unique_instance
