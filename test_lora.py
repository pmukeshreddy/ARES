import torch
import torch.nn as nn

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(10, 10))
    def forward(self, x):
        return x @ self.lora_A

model = DummyModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)

# Save SFT
sft_state = model.lora_A.data.clone()

x = torch.randn(1, 10)
y = torch.randn(1, 10)

# Microbatch 1
optimizer.zero_grad()
current = model.lora_A.data.clone()
model.lora_A.data.copy_(sft_state)
# Ref pass
ref_out = model(x)
model.lora_A.data.copy_(current)

# Curr pass
curr_out = model(x)
loss = ((curr_out - y)**2).mean()
loss.backward()
print("Grad norm before step:", model.lora_A.grad.norm().item())

optimizer.step()
print("Norm after step:", model.lora_A.norm().item())

# Next step
model.lora_A.data.copy_(sft_state)
print("Norm after SFT copy:", model.lora_A.norm().item())
