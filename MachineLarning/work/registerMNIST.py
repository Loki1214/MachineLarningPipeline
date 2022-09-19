# %%
# %cd /home/jovyan/work

import setproctitle
setproctitle.setproctitle('registerMNIST')

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
import os
import multiprocessing
workers = multiprocessing.cpu_count()

import boto3
from botocore.exceptions import ClientError
from botocore.config     import Config
s3 = boto3.resource(
		service_name          = "s3",
		endpoint_url          = "http://minio:9000",
		aws_access_key_id     = os.getenv('MINIO_ROOT_USER'),
		aws_secret_access_key = os.getenv('MINIO_ROOT_PASSWORD'),
		config                = Config(max_pool_connections=workers,
									   proxies={'http':  os.getenv('HTTP_PROXY'),
									   			'https': os.getenv('HTTPS_PROXY')})
)
try:
	bucket = s3.create_bucket(Bucket='digit-images')
except ClientError as e:
	if e.response['Error']['Code'] in ('BucketAlreadyExists', 'BucketAlreadyOwnedByYou'):
		bucket = s3.Bucket('digit-images')
		bucket.objects.all().delete()
	else:
		print(f'Unknown exception.\n\t ' + e.response['Error']['Code'])
		raise

# %%
import MySQLdb
mysql = MySQLdb.connect(
		user     = os.getenv('MYSQL_USER'),
		password = os.getenv('MYSQL_PASSWORD'),
		database = os.getenv('MYSQL_DATABASE'),
		host     = os.getenv('DATABASE'),
		port     = 3306
)
cursor = mysql.cursor()
cursor.execute(f'drop table IF EXISTS MNIST')
cursor.execute(f'create table IF NOT EXISTS \
	MNIST(id INT AUTO_INCREMENT primary key, relpath varchar(100), label INT, date DATETIME, is_used BOOLEAN)')
mysql.commit()
mysql.close()

# %%
import io
from multiprocessing import Pool, Value, Array

def init():
	global s3, bucket
	s3 = boto3.resource(
		service_name          = "s3",
		endpoint_url          = "http://minio:9000",
		aws_access_key_id     = os.getenv('MINIO_ROOT_USER'),
		aws_secret_access_key = os.getenv('MINIO_ROOT_PASSWORD'),
		config                = Config(max_pool_connections=workers,
									   proxies={'http':  os.getenv('HTTP_PROXY'),
									   			'https': os.getenv('HTTPS_PROXY')})
	)
	bucket = s3.Bucket('digit-images')

	global mysql, mysqlCursor #接続オブジェクトをグローバル変数で定義する。
	mysql = MySQLdb.connect(
		user     = os.getenv('MYSQL_USER'),
		password = os.getenv('MYSQL_PASSWORD'),
		database = os.getenv('MYSQL_DATABASE'),
		host     = os.getenv('DATABASE'),
		port     = 3306
	)
	mysqlCursor = mysql.cursor()

indices  = Array('i', [0]*10)
uploaded = Value('i', 0)
def register_mnist_images(idx, dataset, prefix) -> None:
	image, label = dataset[idx]
	with indices.get_lock():
		subIndex = indices[label]
		indices[label] += 1
		uploaded = sum(indices[:])
		print(f'\t{uploaded} / {len(dataset)}', end="")
	basename = prefix + f'_{label}_{subIndex}.jpg'

	image_bytes = io.BytesIO()
	image.save(image_bytes, format="JPEG")
	image_bytes.seek(0)
	bucket.upload_fileobj(Fileobj=image_bytes, Key='MNIST/'+basename)

	mysqlCursor.execute(f"INSERT INTO MNIST(relpath,label,date,is_used) \
		VALUES('{basename}',{label},NULL,false)")
	mysql.commit()

class register_mnist_images_wrapper:
	def __init__ (self, dataset, prefix):
		self.dataset = dataset
		self.prefix  = prefix

	def __call__ (self, index):
		register_mnist_images(index, self.dataset, self.prefix)
		print('')
		return None

# %%
import shutil
if __name__ == '__main__':
	with Pool(processes=workers, initializer=init) as parallel:
		indices[:]  = [0]*10
		uploaded.value = 0
		print('Uploading training data ...')
		parallel.map(register_mnist_images_wrapper(train_dataset, prefix='train'), range(0,len(train_dataset)))
		print(indices[:], uploaded.value)
		print(sum(indices))

		indices[:]  = [0]*10
		uploaded.value = 0
		print('Uploading test data ...')
		parallel.map(register_mnist_images_wrapper(test_dataset, prefix='test'), range(0,len(test_dataset)))
		print(indices[:], uploaded.value)
		print(sum(indices))
		parallel.close()
	shutil.rmtree('data/torch')
