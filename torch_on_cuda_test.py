import torch
import torch.optim
import torch.nn as nn

print("PyTorch version:", torch.__version__)
print("CUDA is available:", torch.cuda.is_available())
print("Number of CUDA devices:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())

if torch.cuda.is_available():
    tensor = torch.rand(3, 3)
    tensor_gpu = tensor.to("cuda")
    print("Tensor on CUDA:", tensor_gpu)


class IntensiveModel(nn.Module):
    def __init__(self):
        super(IntensiveModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))
        self.fc1 = nn.Linear(256 * 5 * 5, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = x.view(-1, 256 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = IntensiveModel().cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.CrossEntropyLoss()

data = torch.randn(64, 3, 64, 64).cuda()
labels = torch.randint(0, 10, (64,)).cuda()

for _ in range(100):
    optimizer.zero_grad()
    preds = model(data)
    loss = loss_func(preds, labels)
    loss.backward()
    optimizer.step()

print("Done")
