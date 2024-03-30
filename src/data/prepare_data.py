import sys, logging, argparse, os
from pathlib import Path

import utils


def process_args(args):
    parser = argparse.ArgumentParser(description='Download processed data and generate vocab')

    parser.add_argument('--dataset-dir', dest='dataset_dir',
                        type=str, required=True,
                        help=('A directory to download the dataset into'
                        ))
    parser.add_argument('--label-file', dest='label_file',
                        type=str, required=True,
                        help=('Input file containing a tokenized formula per line.'
                        ))
    parser.add_argument('--vocab-file', dest='vocab_file',
                        type=str, required=True,
                        help=('Vocab file for putting the vocabulary.'
                        ))
    parser.add_argument('--unk-threshold', dest='unk_threshold',
                        type=int, default=1,
                        help=('If the number of occurences of a token is less than (including) the threshold, then it will be excluded from the generated vocabulary.'
                        ))
    parser.add_argument('--log-path', dest="log_path",
                        type=str, default='log.txt',
                        help=('Log file path, default=log.txt' 
                        ))
    parameters = parser.parse_args(args)
    return parameters




# BASE_DIR = Path(__file__).resolve().parents[1]
# print(BASE_DIR)
# DATASET_DIR = BASE_DIR / "dataset"
# print(DATASET_DIR)
# PROCESSED_IMGS_DIR = DATASET_DIR / "formula_images_processed"
# PROCESSED_IMAGES_DIRNAME = DATA_DIRNAME / "formula_images_processed"
# VOCAB_FILE = PROJECT_DIRNAME / "image_to_latex" / "data" / "vocab.json"


def main(args):

    parameters = process_args(args)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename=parameters.log_path)
    

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.info('Script started: %s'%__file__)

    # im2latex data from https://im2markup.yuntiandeng.com/data/
    # this dataset has been processed and is ready for train and test
    DATASET_DICT = {
        "im2latex_formulas.norm.lst": "https://im2markup.yuntiandeng.com/data/im2latex_formulas.norm.lst",
        "im2latex_formulas.tok.lst": "https://im2markup.yuntiandeng.com/data/im2latex_formulas.tok.lst",
        "formula_images_processed.tar.gz": "https://im2markup.yuntiandeng.com/data/formula_images_processed.tar.gz",
        "im2latex_train_filter.lst": "https://im2markup.yuntiandeng.com/data/im2latex_train_filter.lst",
        "im2latex_validate_filter.lst": "https://im2markup.yuntiandeng.com/data/im2latex_validate_filter.lst",
        "im2latex_test_filter.lst": "https://im2markup.yuntiandeng.com/data/im2latex_test_filter.lst",
    }

    data_dir = parameters.dataset_dir
    os.makedirs(data_dir, exist_ok=True)
    # data_dir.mkdir(parents=True, exist_ok=True)

    train_path = "".join([data_dir, "/im2latex_train_filter.lst"])
    label_path = "".join([data_dir, "/", parameters.label_file])
    vocab_file = "".join([data_dir, "/", parameters.vocab_file])

    # download dataset files
    for filename, url in DATASET_DICT.items():
        filepath = "".join([data_dir, "/", filename])
        if not Path(filepath).is_file():
            utils.download_from_url(url, filepath)
    logging.info("Dataset files downloaded")

    # Untar Processed images
    formula_images_filename = "".join([data_dir, "/formula_images_processed.tar.gz"])
    utils.extract_tarfile(formula_images_filename, data_dir)
    # if not PROCESSED_IMGS_DIR.exists():
        # PROCESSED_IMGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.info("Image files unzipped")
    

    formulas = open(label_path).readlines()
    vocab = {}
    max_len = 0
    with open(train_path) as f:
        for line in f:
            _, line_idx = line.strip().split()
            line_strip = formulas[int(line_idx)].strip()
            tokens = line_strip.split()
            tokens_out = []
            for token in tokens:
                tokens_out.append(token)
                if token not in vocab:
                    vocab[token] = 0
                vocab[token] += 1

    vocab_sort = sorted(list(vocab.keys()))
    vocab_out = []
    num_unknown = 0
    for word in vocab_sort:
        if vocab[word] > parameters.unk_threshold:
            vocab_out.append(word)
        else:
            num_unknown += 1
    vocab = [word for word in vocab_out]

    with open(vocab_file, 'w') as f:
        f.write('\n'.join(vocab))
    logging.info('No. unknowns: %d'%num_unknown)
        

if __name__ == "__main__":
    main(sys.argv[1:])
    logging.info('Script finished')
    