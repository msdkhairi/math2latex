
from utils import LitMathToLatex, LitMathToLatexDataModule
from config import config

from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner

def main():
    # runner = Runner(config)
    # runner.train()

    datamodule = LitMathToLatexDataModule(config)
    model = LitMathToLatex(config)

    logger = TensorBoardLogger(save_dir="logs/", name="math2latex")

    trainer = Trainer(
        logger=logger,
        max_epochs=config.trainer.epochs,
        accelerator=config.trainer.accelerator,
        enable_progress_bar=True
    )

    # tuner = Tuner(trainer=trainer)

    # tuner.scale_batch_size(model, datamodule=datamodule, mode="power")
    # tuner.lr_find(model, datamodule=datamodule)

    trainer.fit(model, datamodule=datamodule)



if __name__ == '__main__':
    main()