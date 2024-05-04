
from utils import LitMathToLatex, LitMathToLatexDataModule
from config import config

from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

def main():
    # runner = Runner(config)
    # runner.train()

    datamodule = LitMathToLatexDataModule(config)
    model = LitMathToLatex(config)

    logger = TensorBoardLogger(save_dir="logs/", name="math2latex", )

    trainer = Trainer(
        logger=logger,
        max_epochs=config.trainer.epochs,
        accelerator=config.trainer.accelerator,
        enable_progress_bar=True
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.fit()



if __name__ == '__main__':
    main()