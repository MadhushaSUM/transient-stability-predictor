import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
from agent.agent import Agent
from functions import *

@tf.keras.utils.register_keras_serializable()
def mean_squared_error(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

# Check if the correct arguments are passed
if len(sys.argv) != 3:
    print("Usage: python evaluate.py [stock] [model]")
    exit()

stock_name, model_name = sys.argv[1], sys.argv[2]

# Load model with custom MSE loss function
model = load_model("models/" + model_name, custom_objects={'mse': mean_squared_error})

# Ensure model has the correct input shape
if len(model.layers[0].input.shape) > 1:
    window_size = model.layers[0].input.shape[1]
else:
    raise ValueError("Model input shape does not have expected dimensions.")

# Initialize the agent
agent = Agent(window_size, True, model_name)

# Load the stock data
data = getStockDataVec(stock_name)
l = len(data) - 1
batch_size = 32  # Defined but not used; might be useful in further steps.

# Initialize the first state and other variables
state = getState(data, 0, window_size + 1)
total_profit = 0
agent.inventory = []

# Main loop over stock data
for t in range(l):
    action = agent.act(state)

    # Get the next state and assume reward = 0 initially
    next_state = getState(data, t + 1, window_size + 1)
    reward = 0

    # Buy action
    if action == 1:
        agent.inventory.append(data[t])
        print("Buy: " + formatPrice(data[t]))

    # Sell action
    elif action == 2 and len(agent.inventory) > 0:
        bought_price = agent.inventory.pop(0)
        reward = max(data[t] - bought_price, 0)
        total_profit += data[t] - bought_price
        print("Sell: " + formatPrice(data[t]) + " | Profit: " + formatPrice(data[t] - bought_price))

    # Determine if the episode is done
    done = True if t == l - 1 else False

    # Append the experience to memory
    agent.memory.append((state, action, reward, next_state, done))
    state = next_state

    # Output profit at the end
    if done:
        print("--------------------------------")
        print(stock_name + " Total Profit: " + formatPrice(total_profit))
        print("--------------------------------")
