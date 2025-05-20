import matplotlib.pyplot as plt
from IPython import display
import numpy as np
import os

plt.ion()

folder_name = "plots"
# os.makedirs(folder_name, exist_ok=True)

def training_process_plot(scores, mean_scores, filename):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

    # Save the plot every time it's updated
    if filename != "":
        print("SAVED GRAPH!")
        plt.savefig(os.path.join(folder_name, filename))


def generate_plots(train_scores, train_mean_scores, test_scores, test_mean_scores, epsilon_values): 
    """Generates all the plots for training and testing."""
    pass

    # Plotting training and testing rewards
    plot_train(train_scores, train_mean_scores, filename="train_scores.png")
    plot_test(test_scores, test_mean_scores, filename="test_scores.png")
    
    # Plotting epsilon decay
    plot_epilson(epsilon_values, filename="epsilon_decay.png")

    # Plotting loss function
    # plot_loss(loss_values, filename="loss_function.png")

    # Plotting Q-values
    # plot_q_values(q_values, filename="q_values.png")


def plot_train(train_scores, train_mean_scores, filename=""):
    plt.figure(figsize=(10, 6))

    plt.title("Training Reward per Trial")
    plt.xlabel("Trials")
    plt.ylabel("Rewards")
    plt.plot(train_scores, color="darkviolet", label="Train Scores")
    plt.plot(train_mean_scores, color="orange", label="Mean Train Scores")
    plt.legend()
    plt.grid(True)


    if filename:
        plt.savefig(os.path.join(folder_name, filename))

    plt.show()
    

def plot_test(test_scores, test_mean_scores, filename=""):
    """Plots training and testing rewards in a single figure."""

    plt.figure(figsize=(10, 6))

    plt.title("Testing Reward at Trials")
    plt.xlabel("Trials")
    plt.ylabel("Rewards")
    plt.plot(test_scores, color="violet", label="Test Scores")
    plt.plot(test_mean_scores, color="goldenrod", label="Mean Test Scores")
    plt.legend()
    plt.grid(True)

    # SAVING PLOT
    if filename:
        plt.savefig(os.path.join(folder_name, filename))

    plt.show()

def plot_epilson(epsilon_values, filename=""):
    """Plots epsilon decay over training episodes."""

    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_values, color="green", linewidth=2)
    plt.title("Epsilon Decay Over Trials")
    plt.xlabel("Trials")
    plt.ylabel("Epsilon")
    plt.grid(True)

    if filename:
        plt.savefig(os.path.join(folder_name, filename))
    
    plt.show()


def plot_loss(loss_values, filename=""):
    """Plots loss function values over training iterations."""

    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, color="purple", linewidth=2)
    plt.title("Loss Over Training Iterations")
    plt.xlabel("Training Iteration")
    plt.ylabel("Loss")
    plt.grid(True)

    if filename:
        plt.savefig(os.path.join(folder_name, filename))

    plt.show()

def plot_q_values(q_values, filename=""):
    """Plots mean Q-values per episode to track stability."""

    pass

