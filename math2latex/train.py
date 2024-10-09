import torchtext; torchtext.disable_torchtext_deprecation_warning()

# from utils import LitMathToLatex, LitMathToLatexDataModule
from runner import LitMathToLatex, LitMathToLatexDataModule
from config import config as cfg

from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner

def main():

    datamodule = LitMathToLatexDataModule(**cfg.datamodule.to_dict())
    model = LitMathToLatex(**cfg.litmodel.to_dict())

    logger = TensorBoardLogger(save_dir="logs/", name="math2latex")

    trainer = Trainer(
        logger=logger,
        **cfg.trainer.to_dict()
    )

    # tuner = Tuner(trainer=trainer)
    # tuner.scale_batch_size(model, datamodule=datamodule, mode="power")
    # tuner.lr_find(model, datamodule=datamodule)

    trainer.fit(model, datamodule=datamodule)

if __name__ == '__main__':
    main()
