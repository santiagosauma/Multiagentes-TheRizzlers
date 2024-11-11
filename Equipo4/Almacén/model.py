import os

from mesa.model import Model
from .agent import Box, Bot, Goal

from mesa.space import SingleGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector


class Maze(Model):
    DEFAULT_MODEL_DESC = [
        'BBBBBBBBBBBBBBBBBBBB',
        'B1FFFFFFFFFFFFFFFBFB',
        'BBFFFFFBBBBBBFFFFBFB',
        'BFFBBBBBFFFFBFFBBBFB',
        'BFFFFBBBFBBBBFFFBFFB',
        'BBBFBBFFFBBFFFFBBFFB',
        'BFFBBFFBFFFFBBFFBBFB',
        'BBFBFFFFBBBFFBBFFFFB',
        'BFBBFFFFFBGFBBFFFFFB',
        'BFFFFFFBBFFFFFBBBBBB',
        'BFFFFBBFFBBBBBBFFFFB',
        'BBFBBFFFFFBBFFFFBBBB',
        'BBFFFFBBFFFFBBFFBBFB',
        'BFFBBBFFFFFBFFFFFBFB',
        'BFBFFBFFFFFFFBBFFFFB',
        'BFBBFFBFFFFFBBFBBBBB',
        'BFFFFFBBFFFFBBFFFFFB',
        'BFBFBFBFBBFFFFFFBBFB',
        'B2FFFFFFBBFFFFFFFFBB',
        'BBBBBBBBBBBBBBBBBBBB'
    ]

    def __init__(self, desc_file=None, **kwargs):
        super().__init__()
        self.goal_states = []

        self.enable_decay = kwargs.get("enable_decay", False)

        if desc_file is None or desc_file == "None":
            desc = self.DEFAULT_MODEL_DESC
        else:
            root_path = os.path.dirname(os.path.abspath(__file__))
            desc = self.from_txt_to_desc(os.path.join(root_path, "mazes", desc_file))

        M, N = len(desc), len(desc[0])

        self.grid = SingleGrid(N, M, False)

        num_bots = 0
        for i in range(M):
            for j in range(N):
                if desc[i][j].isdigit():
                    num_bots += 1
        self.num_bots = num_bots
        self.bots = {}

        self.train_episodes = kwargs.get("train_episodes", 1000)
        self.alpha = kwargs.get("alpha", 0.1)
        self.gamma = kwargs.get("gamma", 0.9)
        self.epsilon = kwargs.get("epsilon", 0.1)

        for i in range(self.num_bots):
            q_file = kwargs.get(f"q_file_bot{i+1}", None)
            setattr(self, f"q_file_bot{i+1}", q_file)

            train_bot = kwargs.get(f"train_bot{i+1}", True)
            setattr(self, f"train_bot{i+1}", train_bot)

        self.schedule = SimultaneousActivation(self)

        self.place_agents(desc)

        self.states = {}
        self.rewards = {}
        for state, cell in enumerate(self.grid.coord_iter()):
            a, pos = cell

            self.states[pos] = state

            if isinstance(a, Goal):
                self.rewards[state] = 1
                self.goal_states.append(state)
            elif isinstance(a, Box):
                self.rewards[state] = -1
            else:
                self.rewards[state] = 0

        model_reporters = {
            f"Bot{i+1}": lambda m, i=i: m.bots[i+1].total_return / (m.bots[i+1].movements + 1)
            if m.bots[i+1].movements > 0 else 0
            for i in range(self.num_bots)
        }

        self.datacollector = DataCollector(
            model_reporters=model_reporters,
        )

    def step(self):
        for bot_id, bot in self.bots.items():
            if self.__getattribute__(f"train_bot{bot_id}"):
                bot.train(episodes=self.train_episodes, alpha=self.alpha, gamma=self.gamma)
                self.__setattr__(f"train_bot{bot_id}", False)

        self.datacollector.collect(self)
        self.schedule.step()

        self.running = not any([a.done for a in self.schedule.agents])

    def place_agents(self, desc: list):
        for y in range(self.grid.height):
            for x in range(self.grid.width):
                cell_value = desc[self.grid.height - y - 1][x]
                pos = (x, y)
                if cell_value == 'B':
                    box = Box(int(f"1000{x}{y}"), self)
                    self.grid.place_agent(box, pos)
                elif cell_value == 'G':
                    goal = Goal(int(f"10{x}{y}"), self)
                    self.grid.place_agent(goal, pos)
                else:
                    try:
                        bot_num = int(cell_value)
                        q_file = eval(f"self.q_file_bot{bot_num}")
                        bot = Bot(int(f"{bot_num}"), self, q_file, self.epsilon)
                        self.grid.place_agent(bot, pos)
                        self.schedule.add(bot)
                        self.bots[bot_num] = bot
                    except ValueError:
                        pass

    @staticmethod
    def from_txt_to_desc(file_path):
        try:
            with open(file_path, 'r') as file:
                desc = [line.rstrip('\n') for line in file]
            return desc
        except Exception as e:
            print(f"Error reading the file: {e}")
            return None