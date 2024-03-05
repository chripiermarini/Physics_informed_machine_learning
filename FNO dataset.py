from neuralop.models import FNO
import torch
from neuralop.datasets import load_darcy_flow_small


device = 'cpu'
train_loader, test_loaders, data_processor = load_darcy_flow_small(
        n_train=1000, batch_size=32,
        test_resolutions=[16, 32], n_tests=[100, 50],
        test_batch_sizes=[32, 32],
        positional_encoding=True
)

data_processor = data_processor.to(device)
model = FNO(n_modes=(16, 16), hidden_channels=64,
                in_channels=3, out_channels=1)
model = model.to(device)

train_sample = train_loader.dataset
data = train_sample[0] ##single sample
data = data_processor.preprocess(data, batched=False)
input = data['x']

nu = input[0].requires_grad_(True)
x1 = input[1].requires_grad_(True)
x2 = input[2].requires_grad_(True)

out = model(torch.stack((nu, x1, x2), 0).unsqueeze(0)).requires_grad_(True)

first_derivative = torch.autograd.grad(outputs=out.sum(),
                                                 inputs= (x1, x2),
                                                 create_graph=True,
                                       allow_unused= True)

print(first_derivative)