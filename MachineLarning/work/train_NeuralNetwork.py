# %%
import setproctitle
setproctitle.setproctitle('train_NeuralNetwork')

# %%
import torch
# GPU(CUDA)が使えるかどうか？
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.deterministic = False  # 非決定論的である代わりに高速化
torch.backends.cudnn.benchmark = True       # 画像サイズが変化しない場合に高速化

# %% [markdown]
# # Import data list from the database

# %%
import os, sys, csv
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.sql import select
from database.settings import DBManager
from database.tables   import MNIST, Uploaded

db = DBManager()
engine = create_engine(f"{db.dialect}://{db.username}:{db.password}@{db.host}:{db.port}/{db.database}?charset=utf8")

# %%
import pandas as pd
newdata_filename='./data/newdata_list.csv'
try:
	df = pd.read_sql(sql=select(Uploaded), con=engine)
	df['relpath'] = f"{Uploaded.__tablename__}/" + df['relpath']

except Exception as err:
	if "Table" and "doesn't exist" in err.args[0]:
		print(f"No data in {db.database}/{Uploaded.__tablename__}")
	else:
		raise
else:
	if df.shape[0] > 0:
		print(f"Loading data from {db.host}/{Uploaded.__tablename__}")
		os.makedirs(os.path.join('data/', Uploaded.__tablename__), exist_ok=True)

		newId = df[ [ not x for x in df['is_used']] ]
		newId = newId['id']
		newId.to_csv(newdata_filename, header=False, index=False)
		images = df.loc[:,['relpath','label']]
	else:
		print(f"No data in {db.database}/{Uploaded.__tablename__}")

# %%
import pandas as pd
df = pd.read_sql(sql=select(MNIST), con=engine)
df['relpath'] = f"{MNIST.__tablename__}/" + df['relpath']

print(f"Loading data from {db.host}:{db.database}/{MNIST.__tablename__}")
os.makedirs(os.path.join('data/', f'{MNIST.__tablename__}'), exist_ok=True)
if 'images' in locals():
	images = pd.concat([ images, df.loc[:,['relpath','label']] ])
else:
	images = df.loc[:,['relpath','label']]

# %%
# エンジン破棄
engine.dispose()

# %% [markdown]
# # Download data from the storage

# %%
import multiprocessing
from multiprocessing import Pool, Manager
workers = os.cpu_count()

# %%
from storage.settings import StorageManager
import boto3
from botocore.exceptions import ClientError
from botocore.config     import Config

from sqlalchemy.orm import Session


storage = StorageManager()
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


# %%
manager = multiprocessing.Manager()
notFoundList = manager.list()
def download_images(relpath):
	localpath = os.path.join('./data', relpath)
	if not os.path.isfile(localpath):
		try:
			bucket.download_file(Key=relpath, Filename=localpath)
		except ClientError as e:
			if e.response['Error']['Code'] == "404":
				# print(f"The object \"{relpath}\" does not exist.")
				dirname  = os.path.dirname(relpath)
				basename = os.path.basename(relpath)
				entry = session.query(Uploaded).filter_by(relpath=f'{basename}').first()
				session.delete(entry)
				notFoundList.append(relpath)
				session.commit()
			else:
				raise

# %%
if __name__ == '__main__':
	print(f"Downloading images from the data storage...")
	with Pool(processes=workers, initializer=init) as parallel:
		parallel.map(download_images, images.loc[:,'relpath'])
		flag = images['relpath'].isin(list(notFoundList))
		flag = [not x for x in flag]
		images = images[flag]
		csv_filename ='./data/data_list.csv'
		images.to_csv(csv_filename, header=False, index=False)
	print(f"Download completed.")

# %% [markdown]
# # Training the model

# %%
from model_definition import Net
#----------------------------------------------------------
# ニューラルネットワークの生成
model = Net().to(device)

# %%
from custom_dataset import MyDataset
full_dataset = MyDataset(
	csv_file=csv_filename,
	root_dir='./data',
	transform=model.transform
)

# 学習データ、検証データに 8:2 の割合で分割する。
train_size = int(0.8 * len(full_dataset))
test_size  = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
	full_dataset, [train_size, test_size]
)

#----------------------------------------------------------
# 学習用／評価用のデータセットの作成
# ハイパーパラメータなどの設定値
num_epochs = 100       # 学習を繰り返す回数
num_batch = 100        # 一度に処理する画像の枚数
learning_rate = 0.001  # 学習率

# データローダー
train_dataloader = torch.utils.data.DataLoader(
	train_dataset,
	batch_size = num_batch,
	shuffle = True)
test_dataloader = torch.utils.data.DataLoader(
	test_dataset,
	batch_size = num_batch,
	shuffle = True)

# %%
import torch.nn as nn
#----------------------------------------------------------
# 学習
model.train()  # モデルを訓練モードにする

#----------------------------------------------------------
# 損失関数の設定
criterion = nn.CrossEntropyLoss()

#----------------------------------------------------------
# 最適化手法の設定
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# %%
from statistics import mean
print(f"Start training.")
epoch_losses = []
for epoch in range(num_epochs): # 学習を繰り返し行う
	losses = []

	for inputs, labels in train_dataloader:
		# GPUが使えるならGPUにデータを送る
		inputs = inputs.to(device)
		labels = labels.to(device)

		# # optimizerを初期化
		optimizer.zero_grad()

		# # ニューラルネットワークの処理を行う
		outputs = model(inputs)

		# # 損失(出力とラベルとの誤差)の計算
		loss = criterion(outputs, labels)
		losses.append(loss.item())
		# loss_sum += loss

		# # 勾配の計算
		loss.backward()

		# # 重みの更新
		optimizer.step()

	# 学習状況の表示
	epoch_losses.append( mean(losses) )
	print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {epoch_losses[-1]}")
	sys.stdout.flush()

# モデルの重みの保存
torch.save(model.state_dict(), 'model_weights.pth')

# %%
import datetime
t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, "JST")
now = datetime.datetime.now(JST)
os.makedirs('log', exist_ok=True)
with open(now.strftime("log/%Y%m%d%H%M%S_losses.log"), "w") as file:
	file.writelines([ str(x)+'\n' for x in epoch_losses ])

# %% [markdown]
# # Testing the model

# %%
#----------------------------------------------------------
# 評価
model.eval()  # モデルを評価モードにする

losses = []
correct = 0

with torch.no_grad():
	for inputs, labels in test_dataloader:
		# GPUが使えるならGPUにデータを送る
		inputs = inputs.to(device)
		labels = labels.to(device)

		# ニューラルネットワークの処理を行う
		outputs = model(inputs)

		# 損失(出力とラベルとの誤差)の計算
		loss = criterion(outputs, labels)
		losses.append(loss.item())

		# 正解の値を取得
		pred = outputs.argmax(1)
		# 正解数をカウント
		correct += pred.eq(labels.view_as(pred)).sum().item()

print(f"Loss: {mean(losses)}, Accuracy: {100*correct/len(test_dataset)}% ({correct}/{len(test_dataset)})")

# %% [markdown]
# # Updating the database

# %%
if os.path.isfile(newdata_filename):
	with open(newdata_filename, 'r') as newdata_file:
		reader = csv.reader(newdata_file)

		engine = create_engine(f"{db.dialect}://{db.username}:{db.password}@{db.host}:{db.port}/{db.database}?charset=utf8")
		session = Session(engine)
		for id in reader:
			newdata = session.query(Uploaded).filter_by(id=id[0]).first()
			newdata.is_used = True
		session.commit()
		session.close()
		engine.dispose()
