#----------------------------------------------------------
# ニューラルネットワークモデルの定義
import torch
import torch.nn as nn
import torch.nn.functional as Functional
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()

        # 各クラスのインスタンス（入出力サイズなどの設定
        innerDim = 1000
        self.fc1 = nn.Linear(input_size, innerDim)
        self.fc2 = nn.Linear(innerDim, output_size)

    def forward(self, x):
        # 順伝播の設定（インスタンスしたクラスの特殊メソッド(__call__)を実行）
        x = self.fc1(x)
        x = torch.sigmoid(x)
        x = self.fc2(x)
        return Functional.log_softmax(x, dim=1)