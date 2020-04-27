from pathlib import Path

class dataset:
    def __init__(self):
        self.data_dir = Path("data/")

    def _get_data(self, directory):
        x = []
        for file in directory.iterdir():
            x.append(file.read_text())
        return x

    def get_data(self):
        return self.data_dir