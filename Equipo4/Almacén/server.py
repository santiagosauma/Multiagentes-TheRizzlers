import mesa
from .model import Maze, Bot, Box, Goal
import os

BOT_COLORS = ["#4169E1", "#DC143C", "#228B22", "#FFD700", "#FF4500", "#8A2BE2", "#FF1493", "#00FFFF", "#FF69B4",
              "#FFA500"]

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

def agent_portrayal(agent):
    if isinstance(agent, Bot):
        return {"Shape": "circle", "Filled": "false", "Color": BOT_COLORS[agent.unique_id - 1], "Layer": 1, "r": 1.0,
                "text": f"{agent.unique_id}", "text_color": "black"}
    elif isinstance(agent, Box):
        object_emoji = "📦"
        return {"Shape": "rect", "Filled": "true", "Layer": 0, "w": 0.9, "h": 0.9, "text_color": "#2F4F4F",
                "Color": "rgba(112, 66, 20, 0.5)", "text": object_emoji}
    elif isinstance(agent, Goal):
        return {"Shape": "rect", "Filled": "true", "Layer": 0, "w": 1, "h": 1, "text_color": "#2F4F4F",
                "Color": "rgba(0, 255, 0, 0.3)", "text": "️⛳️"}

def get_q_files():
    try:
        files = os.listdir(os.path.join(ROOT_PATH, "q_files"))
        files = [f.split(".")[0] for f in files if f.endswith(".json")]
        return ["None"] + sorted(files)
    except FileNotFoundError:
        return ["None"]

def get_maze_files():
    try:
        files = os.listdir(os.path.join(ROOT_PATH, "mazes"))
        files = [f for f in files if f.endswith(".txt")]
        return files
    except FileNotFoundError:
        return []

def get_num_bots():
    dummy_model = Maze()
    return len(dummy_model.bots)

def get_grid_dimensions():
    maze_choices = get_maze_files()
    default_maze = maze_choices[0] if maze_choices else 'None'
    root_path = os.path.dirname(os.path.abspath(__file__))
    maze_path = os.path.join(root_path, "mazes", default_maze)
    try:
        with open(maze_path, 'r') as file:
            desc = [line.rstrip('\n') for line in file]
        width, height = len(desc[0]), len(desc)
        return width, height
    except Exception as e:
        print(f"Error reading the maze file for dimensions: {e}")
        return 50, 34

grid_width, grid_height = get_grid_dimensions()

grid = mesa.visualization.CanvasGrid(
    agent_portrayal, grid_width, grid_height, 1200, 816
)

chart_charges = mesa.visualization.ChartModule(
    [
        {"Label": f"Bot{i + 1}", "Color": BOT_COLORS[i], "label": f"Avg. Reward Bot{i + 1}"} for i in
        range(get_num_bots())
    ],
    data_collector_name='datacollector',
    canvas_height=150,
    canvas_width=600
)

class AgentScore(mesa.visualization.TextElement):
    def __init__(self):
        pass

    def render(self, model):
        scores = []
        for bot_id in sorted(model.bots):
            bot = model.bots[bot_id]
            score = bot.total_return
            scores.append(f"{bot_id}: {score} pts")
        return "<br>".join(scores)

def model_params():
    params = {}

    maze_choices = get_maze_files()
    default_maze = maze_choices[0] if maze_choices else 'None'

    params["desc_file"] = mesa.visualization.Choice(
        name="Maze",
        choices=maze_choices,
        value=default_maze,
        description="Choose the maze file",
    )

    num_bots = get_num_bots()

    for i in range(num_bots):
        params[f"train_bot{i + 1}"] = mesa.visualization.Checkbox(
            name="Train Bot" + str(i + 1),
            value=False,
            description="Train the agent",
        )

        params[f"q_file_bot{i + 1}"] = mesa.visualization.Choice(
            name="Model Bot" + str(i + 1),
            choices=get_q_files(),
            value='None',
            description="Choose the file with the Q-Table",
        )

    params["epsilon"] = mesa.visualization.Slider(
        name="Epsilon",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.01,
        description="Epsilon for the epsilon-greedy policy",
    )

    params["train_episodes"] = mesa.visualization.Slider(
        name="Train Episodes",
        min_value=1,
        max_value=10000,
        value=200,
        step=100,
        description="Number of training episodes",
    )

    params["alpha"] = mesa.visualization.Slider(
        name="Alpha",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.01,
        description="Learning rate",
    )

    params["gamma"] = mesa.visualization.Slider(
        name="Gamma",
        min_value=0.0,
        max_value=1.0,
        value=0.9,
        step=0.01,
        description="Discount factor",
    )

    params["enable_decay"] = mesa.visualization.Checkbox(
        name="Enable Epsilon Decay",
        value=True,
        description="Enable epsilon decay",
    )

    return params

agent_score = AgentScore()

server = mesa.visualization.ModularServer(
    Maze, [grid, agent_score, chart_charges],
    "Bot Maze!", model_params(), 6969
)