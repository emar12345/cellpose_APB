# log_wrapper.py
import csv
# this is bad practice, but since I can't/don't want to override the original cellpose functions, this is how I will get
# some of the statistics.

import logging
import sys
from datetime import datetime
from pathlib import Path
import re
import matplotlib.pyplot as plt


class CellposeIntercepingHandler(logging.Handler):
    def __init__(self):
        super(CellposeIntercepingHandler, self).__init__()
        self.logs = []

    def emit(self, record):
        log_entry = self.format(record)
        self.logs.append(log_entry)


class IOWrapper:
    def logger_setup(self, log_directory=None):
        if log_directory is None:
            cp_dir = Path.home().joinpath('.cellpose')
        else:
            cp_dir = Path(log_directory)

        cp_dir.mkdir(exist_ok=True)
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = cp_dir.joinpath(current_time + '.log')
        try:
            log_file.unlink()
        except:
            print('creating new log file')

        custom_handler = CellposeIntercepingHandler()

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout),
                custom_handler  # Adding the custom handler to the logger
            ]
        )

        logger = logging.getLogger(__name__)
        logger.info(f'WRITING LOG OUTPUT TO {log_file}')

        return logger, log_file, custom_handler

    def get_training_stats(self, logs):
        training_stats = []
        for log_entry in logs:
            pattern = r'Epoch (\d+), Time (\d+\.\d+)s, Loss (\d+\.\d+), Loss Test (\d+\.\d+), LR (\d+\.\d+)'
            match = re.search(pattern, log_entry)
            if match:
                epoch = int(match.group(1))
                time = float(match.group(2))
                loss = float(match.group(3))
                loss_test = float(match.group(4))
                lr = float(match.group(5))

                variable_list = [epoch, time, loss, loss_test, lr]
                training_stats.append(variable_list)
            else:
                # print("Log entry format does not match the expected pattern.")
                # return None
                pass
        return training_stats

    def plot_training_stats(self, stats, model_name):
        # todo: dear lord where do i start with what is wrong with this.
        # piece of garbage
        path = Path("./data/fig", model_name)
        path.mkdir(exist_ok=True, parents=True)

        epochs = [entry[0] for entry in stats]
        time = [entry[1] for entry in stats]
        loss = [entry[2] for entry in stats]
        loss_test = [entry[3] for entry in stats]
        learning_rate = [entry[4] for entry in stats]
        # Create a list of lists with all the data
        data = list(zip(epochs, time, loss, loss_test, learning_rate))

        # Output as CSV
        csv_file = path / "data.csv"

        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "time", "loss", "loss_test", "learning rate"])
            writer.writerows(data)

        print(f"CSV data has been written to {csv_file}")

        # loss over epoch
        plt.plot(epochs, loss, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(path / "/loss_over_epoch.png")
        plt.show()
        # loss
        plt.plot(epochs, loss_test, label='Loss Test')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Test')
        plt.title('Loss Test over Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(path / "loss_test_over_epoch.png")
        plt.show()
        # both on same graph
        plt.plot(epochs, loss_test, label='Loss Test', color='blue')
        plt.plot(epochs, loss, label='Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss and Loss Test over Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(path / "loss_and_loss_test_over_epoch.png")
        plt.show()
        # learning rate over epoch
        plt.plot(epochs, learning_rate, label='Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate over Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(path / "learning_rate_over_epoch.png")
        plt.show()
        # time over epoch
        plt.plot(epochs, time, label='Time')
        plt.xlabel('Epoch')
        plt.ylabel('Time')
        plt.title('Time (Cumulative) over Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(path / "time_over_epoch.png")
        plt.show()



    def get_model_path(self, logs):
        pattern = r'saving network parameters to (\S+)'
        for log_entry in logs:
            match = re.search(pattern, log_entry)
            if match:
                file_path = Path(match.group(1))
                return file_path
