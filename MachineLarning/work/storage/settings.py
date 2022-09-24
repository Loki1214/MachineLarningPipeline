import os
# NOT thread safe
class StorageManager(object):
	dialect  = 'minio'
	username = os.getenv('MINIO_ROOT_USER')
	password = os.getenv('MINIO_ROOT_PASSWORD')
	host     = os.getenv('STORAGE_HOST')
	port     = 9000
	bucket   = 'digit-images'

	_unique_instance = None
	def __new__(cls, *args, **kargs):
		if cls._unique_instance is None:
			cls._unique_instance = super(StorageManager, cls).__new__(cls)
		return cls._unique_instance