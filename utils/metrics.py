import numpy as np
import matplotlib.pyplot as plt

class TrainingLogger :
    """
    Tracks and aggregates the defender's loss over the training episodes
    to evaluate the robustness of the ARL approach against baselines.
    """
    def __init__(self,case_name = "Intrusion Detection"):
        self.case_name = case_name
        self.episode_losses = []
        self.current_episode_loss = 0.0

    def add_step_loss(self, loss):
        self.current_episode_loss += loss

    def end_episode(self):
        self.episode_losses.append(self.current_episode_loss)
        self.current_episode_loss = 0.0

    def get_average_episode_loss(self,window = 50):
        if len(self.episode_losses) < window:
            return np.mean(self.episode_losses) if self.episode_losses else 0.0
        return np.mean(self.episode_losses[-window:])

    def plot_learning_curve(self, save_path=None):

        if not self.episode_losses:
            print("No data to plot.")
            return

        window = max(1, len(self.episode_losses) // 20)
        moving_avg = np.convolve(self.episode_losses, np.ones(window) / window, mode='valid')

        plt.figure(figsize=(8, 5))
        plt.plot(self.episode_losses, alpha=0.3, color='red', label='Raw Episode Loss')
        plt.plot(np.arange(window - 1, len(self.episode_losses)), moving_avg,
                 color='darkred', linewidth=2, label=f'Moving Avg (n={window})')

        plt.title(f"Defender's Loss over Time ({self.case_name})")
        plt.xlabel("Episode")
        plt.ylabel("Defender's Total Loss")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()



