import numpy as np
import random
import keyboard

class Game:
    def __init__(self, board = None) -> None:
        if board is None: self.board = np.zeros((4,4))
        else: self.board = board.copy().reshape((4,4))
        self.score = 0
        self.done = False


    def generate(self) -> None:
        blanks = []
        for i in range(4):
            for j in range(4):
                if self.board[i,j] == 0: blanks.append((i,j))
        if len(blanks) == 0: return
        p = random.choice(blanks)
        self.board[p[0],p[1]] = 2

    def seed(self, s : int) -> None:
        np.random.seed(s)
        random.seed(s)

    def reset(self) -> np.ndarray:
        self.board = np.zeros((4,4), dtype=int)
        self.score = 0
        self.done = False
        self.generate()
        return self.board.reshape((16,))

    def check(self) -> bool:
        flag = False
        for i in range(4):
            for j in range(4):
                if self.board[i,j] == 0: flag = True

        if flag: return True

        for i in range(4):
            for j in range(3):
                if self.board[i,j] == self.board[i,j+1]:
                    flag = True
        if flag: return True
        
        for i in range(3):
            for j in range(4):
                if self.board[i,j] == self.board[i+1,j]:
                    flag = True
        return flag



    def align(self, arr):
        for i in range(4):
            k = 0
            for j in range(4):
                if arr[i,j] > 0: 
                    arr[i,j], arr[i,k] = arr[i,k], arr[i,j]
                    k += 1

    def step(self, action, isPrint = False):
        reward = 0
        board_t = self.board.copy()
        if action % 2 == 1: board_t = board_t.T
        if action // 2 == 1: board_t = np.flip(board_t, axis=1)

        self.align(board_t)
        
        for i in range(4):
            for j in range(3):
                if board_t[i,j] > 0 and board_t[i,j] == board_t[i,j+1]:
                    board_t[i,j] *= 2
                    reward += board_t[i,j]
                    board_t[i,j+1] = 0

        self.align(board_t)

        if action // 2 == 1: board_t = np.flip(board_t, axis=1)
        if action % 2 == 1: board_t = board_t.T
        punish_flag = False
        if (self.board == board_t).all(): punish_flag = True # 惩罚无效移动
        self.board = board_t
        self.generate()
        self.score += reward
        self.done = not self.check()
        reward = np.log2(reward + 0.9)
        if self.done or punish_flag:
            reward = -10
        if isPrint : self.printBoard()
        return self.board.reshape((16)), reward, self.done

    def printBoard(self):
        print(self.board)
        print("Score: %d   End: %d"%(self.score, self.done))

    def getnpboard(self):
        return self.board.copy().reshape((16,))
        
if __name__=='__main__':
    env = Game()
    env.reset()
    env.printBoard()

    keyboard.add_hotkey('a', env.step, args=(0, True), suppress=True)
    keyboard.add_hotkey('w', env.step, args=(1, True), suppress=True)
    keyboard.add_hotkey('d', env.step, args=(2, True), suppress=True)
    keyboard.add_hotkey('s', env.step, args=(3, True), suppress=True)

    keyboard.wait('ctrl+c')

    keyboard.unhook_all_hotkeys()
