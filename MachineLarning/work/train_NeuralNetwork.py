# %%
# %cd /home/jovyan/work
import torch
# GPU(CUDA)が使えるかどうか？
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

# %%
from model_definition import Net
#----------------------------------------------------------
# ニューラルネットワークの生成
model = Net().to(device)

# %%
import boto3
from botocore.exceptions import ClientError
from botocore.config     import Config
s3 = boto3.resource(
	service_name          = "s3",
	endpoint_url          = "http://minio:9000",
	aws_access_key_id     = 'minioadminuser',
	aws_secret_access_key = 'minioadminpassword',
	config                = Config(proxies={'http': '', 'https': ''}))
bucket = s3.Bucket('digit-images')

# %%
import MySQLdb
mysql = MySQLdb.connect(
		user='root',
		password='root_password',
		host='mysql',
		port=3306,
		database='DigitImages'
	)
mysqlCursor = mysql.cursor()
tableName='uploaded'

# %%
import os, csv
import numpy as np

mysqlCursor.execute(f"SELECT CONCAT('{tableName}_',relpath), label, id, is_used FROM {tableName}")
rows = np.array(mysqlCursor.fetchall())

newdata_filename='./data/newdata_list.csv'
newdata_file = open(newdata_filename, 'w')
row = rows[:,3].astype(int)
csv.writer(newdata_file).writerows(rows[row == np.zeros(row.shape), 2])
newdata_file.close()

csv_filename ='./data/data_list.csv'
csv_file     = open(csv_filename, 'w')
csv_writer   = csv.writer(csv_file)
csv_writer.writerows(rows[:,0:2])
images = rows[:,0:2]


mysqlCursor.execute(f"SELECT CONCAT('MNIST_',relpath), label, id, is_used FROM MNIST")
rows = np.array(mysqlCursor.fetchall())
images = np.vstack([images,rows[:,0:2]])
csv_writer.writerows(images)
csv_file.close()

for relpath, label in images:
	localpath = os.path.join('./data', relpath)
	if not os.path.isfile(localpath):
		try:
			bucket.download_file(Key=relpath, Filename=localpath)
		except ClientError as e:
			if e.response['Error']['Code'] == "404":
				print("The object does not exist.")
			else:
				raise

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
num_epochs = 10         # 学習を繰り返す回数
num_batch = 100         # 一度に処理する画像の枚数
learning_rate = 0.001   # 学習率

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

for epoch in range(num_epochs): # 学習を繰り返し行う
    loss_sum = 0

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
        loss_sum += loss

        # # 勾配の計算
        loss.backward()

        # # 重みの更新
        optimizer.step()

    # 学習状況の表示
    print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss_sum.item() / len(train_dataloader)}")

    # モデルの重みの保存
    torch.save(model.state_dict(), 'model_weights.pth')

# %%
#----------------------------------------------------------
# 評価
model.eval()  # モデルを評価モードにする

loss_sum = 0
correct = 0

with torch.no_grad():
    for inputs, labels in test_dataloader:

        # GPUが使えるならGPUにデータを送る
        inputs = inputs.to(device)
        labels = labels.to(device)

        # ニューラルネットワークの処理を行う
        outputs = model(inputs)

        # 損失(出力とラベルとの誤差)の計算
        loss_sum += criterion(outputs, labels)

        # 正解の値を取得
        pred = outputs.argmax(1)
        # 正解数をカウント
        correct += pred.eq(labels.view_as(pred)).sum().item()

print(f"Loss: {loss_sum.item() / len(test_dataloader)}, Accuracy: {100*correct/len(test_dataset)}% ({correct}/{len(test_dataset)})")

# %%
newdata_file = open(newdata_filename, 'r')
reader = csv.reader(newdata_file)
print(reader)
for id in reader:
	mysqlCursor.execute(f"UPDATE uploaded SET is_used = true WHERE id = {id[0]};")
newdata_file.close()
mysql.commit()
mysql.close()
