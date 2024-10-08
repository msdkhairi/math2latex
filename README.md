## math2latex

math2latex is a tool that converts math images to LaTeX format. It provides a convenient way to convert handwritten or scanned math equations into LaTeX code, making it easier to incorporate them into documents or use them in mathematical typesetting.

### Installation

To use math2latex, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/msdkhairi/math2latex.git
    ```

2. Navigate to the project directory:

    ```bas
    cd math2latex
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Usage

To prepare the data for conversion, run the following command in the root directory of the repository:

```bash
python math2latex/data/prepare_data.py --dataset-dir "dataset"
 ```
  
This will create a dataset folder with all the dataset files inside --dataset-dir directory.

### Training

The configuration for the training process can be found in the `src/config.py` file.
To train the model, run the following command in the root directory of the repository:

```shell
python math2latex/train.py"
```

This will train the model using the dataset and save the trained model in each epoch in the `runs` directory.

### Test

To test the model, run the following command in the root directory of the repository:

```shell
python math2latex/test.py
```

This will test the model using the test dataset and print the BLEU score, Edit Distance and Exact Match of the model.

### Demo

To run the demo, run the following command in the root directory of the repository:

```shell
python demo.py
```
This command runs a demo of the tool, allowing you to see its functionality in action.

### License

math2latex is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
