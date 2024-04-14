import sys, logging, argparse, os
from pathlib import Path


import utils



def process_args(args):
    parser = argparse.ArgumentParser(description='Download processed data and generate vocab')

    parser.add_argument('--dataset-dir', dest='dataset_dir',
                        type=str, default='dataset',
                        help=('A directory to download the dataset into'
                        ))
    parser.add_argument('--label-file', dest='label_file',
                        type=str, default='im2latex_formulas.norm.lst',
                        help=('Input file containing a tokenized formula per line.'
                        ))
    parser.add_argument('--vocab-file', dest='vocab_file',
                        type=str, default='vocab.json',
                        help=('Vocab file for putting the vocabulary.'
                        ))
    parser.add_argument('--processed-imgs-dir', dest='processed_imgs_dir',
                        type=str, default='formula_images_processed',
                        help=('Directory containing the processed images. Default=formula_images_processed_cropped'
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
        # "formula_images_processed.tar.gz": "https://im2markup.yuntiandeng.com/data/formula_images_processed.tar.gz",
        "formula_images.tar.gz": "https://im2markup.yuntiandeng.com/data/formula_images.tar.gz",
        "im2latex_train_filter.lst": "https://im2markup.yuntiandeng.com/data/im2latex_train_filter.lst",
        "im2latex_validate_filter.lst": "https://im2markup.yuntiandeng.com/data/im2latex_validate_filter.lst",
        "im2latex_test_filter.lst": "https://im2markup.yuntiandeng.com/data/im2latex_test_filter.lst",
    }

    data_dir = parameters.dataset_dir
    os.makedirs(data_dir, exist_ok=True)

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
    formula_images_filename = "".join([data_dir, "/formula_images.tar.gz"])
    utils.extract_tarfile(formula_images_filename, data_dir)
    logging.info("Image files unzipped")

    # cleaning the labels
    logging.info('Cleaning labels...')
    # Clean the ground truth file
    cleaned_file = "im2latex_formulas.norm.processed.lst"
    cleaned_file = "".join([data_dir, "/", cleaned_file])
    if not Path(cleaned_file).is_file():
        print("Cleaning data...")
        utils.find_and_replace(label_path, cleaned_file)
    else:
        print("Cleaned data already exists.")

    # preprocess images
    processed_imgs_dir = "".join([data_dir, "/", parameters.processed_imgs_dir])
    os.makedirs(processed_imgs_dir, exist_ok=True)

    dataset_dir = "".join([data_dir, "/formula_images"])
    processed_imgs_dir = "".join([data_dir, "/", parameters.processed_imgs_dir])

    logging.info('Processing images...')
    utils.process_images(dataset_dir, processed_imgs_dir)
    
    logging.info('Images processed')
    logging.info('Script finished')

      

if __name__ == "__main__":
    main(sys.argv[1:])
    
    