import torch
import torch.nn as nn
from collections import OrderedDict

# This reproduces the issue exactly
model = nn.Linear(10, 10, bias=False)
model.weight.data.fill_(1.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

# Save expected starting point
sft_state = model.weight.data.clone()

x = torch.ones(1, 10)
y = torch.zeros(1, 10)

optimizer.zero_grad()
current = model.weight.data.clone()

# This is the exact swap from dapo_trainer
model.weight.data.copy_(sft_state)
_ = model(x)
model.weight.data.copy_(current)

loss = ((model(x) - y)**2).mean()
loss.backward()

print("Grad norm:", model.weight.grad.norm().item()) # Should be non-zero
optimizer.step()

print("Diff from start:", (model.weight.data - sft_state).norm().item())
