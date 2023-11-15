import torch
from envs.hex_env import HexEnv

env = HexEnv()
obs, info = env.reset()

f = torch.tensor(obs["features"], dtype=torch.float32)
n = torch.tensor(obs["next"])
p = torch.tensor(obs["previous"])
t = torch.tensor(obs["twin"])

# repeat data as batch
f = f.repeat(5, 1, 1)
n = n.repeat(5, 1)
p = p.repeat(5, 1)
t = t.repeat(5, 1)

# offset next values
skip = n.shape[-1]
on = []
offset = 0
for v in n:
    v = v.clone()
    v[v >= 0] += offset
    on.append(v)
    offset += skip

offset_n = torch.cat(on, dim=0)

f = torch.arange(0, 40)
f = f.reshape(2,5,4)
g = f.reshape(-1,4)
h = g.reshape(2,-1,4)