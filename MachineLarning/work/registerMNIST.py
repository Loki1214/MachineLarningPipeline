# %%
import os, setproctitle
setproctitle.setproctitle(os.path.basename(__file__))
workers = os.cpu_count()

# %%
from torchvision import datasets
# MNISTデータの取得
# https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST
# 学習用
train_dataset = datasets.MNIST(
    './data/torch',               # データの保存先
    train = True,           # 学習用データを取得する
    download = True,        # データが無い時にダウンロードする
    # transform = transform   # テンソルへの変換など
    )
# 評価用
test_dataset = datasets.MNIST(
    './data/torch',
    train = False,
    # transform = transform
    )

# %%
from storage.settings import StorageManager
import boto3
from botocore.exceptions import ClientError
from botocore.config     import Config

storage = StorageManager()
s3 = boto3.resource(
		service_name          = "s3",
		endpoint_url          = f"http://{storage.host}:{storage.port}",
		aws_access_key_id     = storage.username,
		aws_secret_access_key = storage.password,
		config                = Config(max_pool_connections=workers,
									   proxies={'http':  os.getenv('HTTP_PROXY'),
									   			'https': os.getenv('HTTPS_PROXY')})
	)
try:
	bucket = s3.create_bucket(Bucket=storage.bucket)
except ClientError as e:
	if e.response['Error']['Code'] in ('BucketAlreadyExists', 'BucketAlreadyOwnedByYou'):
		bucket = s3.Bucket(storage.bucket)
		bucket.objects.all().delete()
	else:
		print(f'Unknown exception.\n\t ' + e.response['Error']['Code'])
		raise

# %%
from sqlalchemy import create_engine, inspect
from sqlalchemy.sql import select
from database.settings import DBManager
from database.tables   import MNIST, Uploaded

db = DBManager()
engine = create_engine(f"{db.dialect}://{db.username}:{db.password}@{db.host}:{db.port}/{db.database}?charset=utf8")
if inspect(engine).has_table(MNIST.__tablename__):
	MNIST.__table__.drop(engine)
MNIST.__table__.create(bind = engine)
engine.dispose()

# %%
import io
from multiprocessing import Pool, Array

from sqlalchemy.orm import Session

def init():
	global bucket
	s3 = boto3.resource(
		service_name          = "s3",
		endpoint_url          = f"http://{storage.host}:{storage.port}",
		aws_access_key_id     = storage.username,
		aws_secret_access_key = storage.password,
		config                = Config(max_pool_connections=workers,
									   proxies={'http':  os.getenv('HTTP_PROXY'),
									   			'https': os.getenv('HTTPS_PROXY')})
	)
	bucket = s3.Bucket(storage.bucket)

	global session
	engine  = create_engine(f"{db.dialect}://{db.username}:{db.password}@{db.host}:{db.port}/{db.database}?charset=utf8")
	session = Session(autocommit=False,
					   autoflush=True,
					   expire_on_commit=False,
					   bind=engine)

indices  = Array('i', [0]*10)
def register_mnist_images(idx, dataset, prefix) -> None:
	image, label = dataset[idx]
	with indices.get_lock():
		subIndex = indices[label]
		indices[label] += 1
		uploaded = sum(indices[:])
		print(f'\t{uploaded} / {len(dataset)}', end='')
	basename = prefix + f'_{label}_{subIndex}.jpg'

	image_bytes = io.BytesIO()
	image.save(image_bytes, format="JPEG")
	image_bytes.seek(0)
	bucket.upload_fileobj(Fileobj=image_bytes, Key=MNIST.__tablename__+'/'+basename)

	session.add( MNIST(relpath=basename, label=label, date=None, is_used=False) )
	session.commit()

class register_mnist_images_wrapper:
	def __init__ (self, dataset, prefix):
		self.dataset = dataset
		self.prefix  = prefix

	def __call__ (self, index):
		register_mnist_images(index, self.dataset, self.prefix)
		print('')
		return None

# %%
if __name__ == '__main__':
	with Pool(processes=workers, initializer=init) as parallel:
		indices[:]  = [0]*10
		print('Uploading training data ...')
		parallel.map(register_mnist_images_wrapper(train_dataset, prefix='train'), range(0,len(train_dataset)))
		print(indices[:])
		print(sum(indices))

		indices[:]  = [0]*10
		print('Uploading test data ...')
		parallel.map(register_mnist_images_wrapper(test_dataset, prefix='test'), range(0,len(test_dataset)))
		print(indices[:])
		print(sum(indices))

# 2m 46s
# 1m 7.8s
