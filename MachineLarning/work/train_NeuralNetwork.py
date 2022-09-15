# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# %%
# GPU(CUDA)が使えるかどうか？
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

# %%
from model_definition import Net
#----------------------------------------------------------
# ニューラルネットワークの生成
model = Net().to(device)

# %%
from custom_dataset import MyDataset
full_dataset = MyDataset(
	csv_file='./data/MNIST.csv',
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
