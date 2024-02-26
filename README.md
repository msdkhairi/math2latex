# math2latex
Convert math images to latex

To prepare data run the following command while in the root directory of the repo:
python data/prepare_data.py --dataset-dir "./dataset" --label-file "im2latex_formulas.norm.lst" --vocab-file "vocab.txt"
This will create a dataset folder with all the dataset files inside. It will also a create a vocab.txt file which contains the vocabulary of the trainset. 