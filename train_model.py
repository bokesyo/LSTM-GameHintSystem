import torch.nn as nn

class LSTMNet(nn.Module):

    def __init__(self, input_size):
        super(LSTMNet, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=100,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Sequential(
            nn.Linear(100, 2)
        )

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
        out = self.out(r_out[:, -1])
        return out


net = LSTMNet(features)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.001)
# start training
for e in range(1000):
    var_x = Variable(X)
    var_y = Variable(y)
    # forward
    out = net(var_x)
    loss = criterion(out, var_y)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))
    if (e + 1) % 100 == 0:  # 每 100 次输出结果
        torch.save(obj=net.state_dict(), f='lstmnet_%d.pth' % (e + 1))
 
torch.save(obj=net.state_dict(), f="lstmnet.pth")



