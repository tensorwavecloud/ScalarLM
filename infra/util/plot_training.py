import json
import matplotlib.pyplot as plt
import argparse
import logging

logger = logging.getLogger(__name__)

# Extract steps and loss
def extract_steps_and_loss(file_path):
    steps = []
    losses = []
    with open(file_path, 'r') as file:
        try:
            data = json.load(file)
            steps = [entry['step'] for entry in data['history']]
            losses = [entry['loss'] for entry in data['history']]
        except json.JSONDecodeError:
           print(f"Invalid json file {file_path}.")

    return steps, losses

def plot_training():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Plot training loss from a JSON log file.')

    # Accept up to 5 log file paths
    for i in range(5):
        parser.add_argument(f'--log{i}', type=str, help='Path to json log file')

    args = parser.parse_args()
    colors = ['green', 'blue', 'red', 'magenta', 'cyan']
    # Create the plot
    plt.figure(figsize=(10, 6))

    i = 0
    for arg, value in vars(args).items():
        if value is None:
            continue
        # Extract steps and losses
        steps, losses = extract_steps_and_loss(value)
        if len(steps) == 0 or len(losses) == 0:
            logger.error("No steps or loss values in the log file.")
            return

        label = value.split(".")[0]
        plt.plot(steps, losses, marker='o', color=colors[i], label=label)
        i+=1
        plt.text(steps[0], losses[0], f'({steps[0]}, {losses[0]:.3f})',
            verticalalignment='bottom')
        plt.text(steps[-1], losses[-1], f'({steps[-1]}, {losses[-1]:.3f})',
            verticalalignment='top')

    # Customize the plot
    plt.title('Training Loss over Steps')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Improve the layout
    plt.tight_layout()

    # Save the plot (optional)
    # plt.savefig('train_loss.png')

    # Show the plot
    plt.show()

plot_training()
