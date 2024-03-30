from .runner import Runner
from .config import config

def main():
    runner = Runner(config)
    runner.train()

if __name__ == '__main__':
    main()