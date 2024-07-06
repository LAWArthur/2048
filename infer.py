from model import Model2048, Model2048Config
import torch
from game import Game
import numpy as np
import time

device = 'cpu'

model = Model2048(Model2048Config())
model.load_state_dict(torch.load('checkpoint.pt'))
model.to(device)
model.eval()

game = Game()
game.reset()

def bestaction(model, curstate):
    with torch.no_grad():
        x = torch.concat((torch.tensor(curstate).unsqueeze(0).repeat(4,1), torch.eye(4)),dim=1).to(device)
        y = model(x).cpu().numpy()
        action = np.argmax(y)
    # print(y)
    return action, y[action]

while not game.done:
    action, _ = bestaction(model, game.getnpboard())
    game.step(action, True)
    time.sleep(0.5)
