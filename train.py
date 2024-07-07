import torch.utils
import torch.utils.data
import torch.utils.data.sampler
from model import Model2048, Model2048Config
import torch
from game import Game
import numpy as np
import torch.nn.functional as F
import copy
import math

device = 'cpu'
if torch.cuda.is_available():
    device = torch.device('cuda')

model = Model2048(Model2048Config())
# model.load_state_dict(torch.load('checkpoint.pt'))
model.compile(backend='eager')
model.to(device)
model.train()
evalmodel: Model2048 = None

epochs = 10000
model_update_steps = 100
eps = 0.999
fall = 0.99999
gamma = 0.5
batch_size = 256


optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)



class Memory:
    def __init__(self, capacity=10000, e=0.01, a=0.6) -> None:
        self.memory = []
        self.weights = []
        self.capacity = capacity
        self.e = e
        self.a = a
    
    def add(self, mem):
        self.memory.append(mem)
        self.weights.append(1e5)
        if len(self.memory) > self.capacity: 
            del self.memory[0]
            del self.weights[0]
    
    def update(self, idx, wht):
        for i in range(len(idx)):
            self.weights[idx[i]] = (np.abs(wht[i]) + self.e) ** self.a

    def sample(self, n):
        return torch.multinomial(torch.tensor(self.weights[:-1]), n, False).numpy()


memory = Memory(capacity=10000)


def bestaction(model, curstate):
    with torch.no_grad():
        x = torch.concat((torch.tensor(curstate).unsqueeze(0).repeat(4,1), torch.eye(4)),dim=1).to(device)
        y = model(x).cpu().numpy()
        action = np.argmax(y)
    # print(y)
    return action, y[action]

def trainstep(step):
    if(len(memory.memory) <= batch_size): return
    indices = memory.sample(batch_size)
    # print(indices)
    x = []
    y = []
    # print(indices)
    for i in indices:
        curstate, action, reward, terminated = memory.memory[i]
        x.append(torch.concat((torch.tensor(curstate), F.one_hot(torch.tensor(action), num_classes=4))))

        if not terminated:
            _,nextreward = bestaction(evalmodel, memory.memory[i+1][0])
            reward += gamma * nextreward
        y.append(reward)
        # print(curstate)
    x = torch.stack(x).type(torch.float32).to(device)
    y = torch.tensor(y).to(device)

    y_pred = model(x)

    errors = (y_pred - y).detach().cpu().numpy()
    memory.update(indices, errors)

    loss = F.mse_loss(y_pred.float(), y.float())
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    print(f'step {step} | loss: {loss.item():.6f} | norm: {norm.item():.4f} | eps: {eps:.5e}')
    # print(x, y)
    # print(memory.weights)

step = 0
for epoch in range(epochs):
    curstate = Game().reset()
    while True:
        

        if step % model_update_steps == 0:
            print('evalmodel updated!')
            evalmodel = copy.deepcopy(model)
            evalmodel.to(device)
            evalmodel.eval()
            torch.save(model.state_dict(), 'checkpoint.pt')

        # -- exploration --
        if np.random.random() > eps:
            action,_ = bestaction(model, curstate)
        else:
            action = np.random.randint(0,4)
        
        newstate, reward, terminated = Game(curstate).step(action)
        memory.add((np.log2(curstate+1, dtype=np.float32), action, reward, terminated))
        curstate = newstate
        eps *= fall

        print(f'step {step} | reward: {reward:.4f}')

        trainstep(step)

        step += 1

        if terminated: break