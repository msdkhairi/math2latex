from PIL import Image
import torch
import torchvision.transforms as transforms

import matplotlib
import matplotlib.pyplot as plt

import gradio as gr

from src.data.utils import get_formulas
from src.data.dataset import TrainDataset, get_dataloader, Tokenizer
from src.model.transformer import ResNetTransformer

# Global variables to hold the setup components
model, tokenizer = None, None

def latex2image(latex_expression, image_name, image_size_in=(3, 0.6), fontsize=12, dpi=200):

    # Runtime Configuration Parameters
    matplotlib.rcParams["mathtext.fontset"] = "cm"  # Font changed to Computer Modern
    # matplotlib.rcParams['text.usetex'] = True  # Use LaTeX to write all text
    # print('hi')

    fig = plt.figure(figsize=image_size_in, dpi=dpi)
    text = fig.text(
        x=0.5,
        y=0.5,
        s=latex_expression,
        horizontalalignment="center",
        verticalalignment="center",
        fontsize=fontsize,
    )

    plt.savefig(image_name)
    plt.close(fig)

def setup():
    global model, tokenizer
    # setup the model
    checkpoint_path = 'runs/model_epoch_25.pth'
    model = ResNetTransformer()
    model.load_state_dict(torch.load(checkpoint_path))
    model.to("cpu")
    model.eval()

    # setup the tokenizer
    formulas = get_formulas('dataset/im2latex_formulas.norm.processed.lst')
    tokenizer = Tokenizer(formulas)
    

def predict_image(image):
    global model, tokenizer
    
    if model is None or tokenizer is None:
        setup()

    transform = transforms.ToTensor()

    image = transform(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        output = model.predict(image)
    
    tokens = tokenizer.decode(output[0].tolist())
    return tokens

def predict_and_convert_to_image(image):

    latex_code = predict_image(image)

    image_name = 'temp.png'
    latex_code_modified = latex_code.replace(" ", "")  # Remove spaces from the LaTeX code
    latex_code_modified = rf"""${latex_code_modified}$"""
    latex2image(latex_code_modified, image_name)
    
    # Return both the LaTeX code and the path of the generated image
    return latex_code, image_name

def main():
    setup()
    # examples = [
    #     ["dataset/formula_images_processed/78228211ca.png"],
    #     ["dataset/formula_images_processed/2b891b21ac.png"],
    #     ["dataset/formula_images_processed/a8ec0c091c.png"],
    # ]
    gr_app = gr.Interface(
        fn=predict_and_convert_to_image, 
        inputs='image', 
        outputs=['text', 'image'],
        # examples=examples,
        title='Image to LaTeX code',
        description='Convert an image of a mathematical formula to LaTeX code and view the result as an image. Upload an image of a formula to get both the LaTeX code and the corresponding image or use the examples provided.'
    )
    gr_app.launch()

if __name__ == "__main__":
    main()
    