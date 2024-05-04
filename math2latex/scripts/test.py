


import torch
import logging
from tqdm import tqdm

from data.utils import get_formulas
from data.dataset import TrainDataset, get_dataloader, Tokenizer
from model.transformer import ResNetTransformer

from nltk.translate.bleu_score import sentence_bleu
from nltk.metrics.distance import edit_distance
from nltk.translate.meteor_score import exact_match



def setup_test():
    # setup the model
    checkpoint_path = 'runs/model_epoch_25.pth'
    model = ResNetTransformer()
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # setup the tokenizer
    formulas = get_formulas('dataset/im2latex_formulas.norm.processed.lst')
    tokenizer = Tokenizer(formulas)

    # setup the test dataloader
    test_dataset = TrainDataset(
        'dataset',
        'formula_images_processed',
        'im2latex_formulas.norm.processed.lst',
        'im2latex_test_filter.lst',
        transform='test',
    )

    test_dataloader = get_dataloader(
        test_dataset,
        tokenizer=tokenizer,
        batch_size=1,
        num_workers=1,
    )

    return model, test_dataloader, tokenizer


def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename='test.log')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.info('Script started: %s'%__file__)

    model, test_dataloader, tokenizer = setup_test()

    logging.info('Model and test dataloader setup complete')

    blue_scores = []
    ed_scores = []
    em_scores = []
    vem_scores = []

    logging.info('Starting test loop')
    pbar = tqdm(test_dataloader, desc='Processing Images')

    for images, targets in pbar:
        target_formula = tokenizer.decode(targets[0].tolist())
        if len(target_formula) == 0:
            continue
        pred = model.predict(images)
        pred_formula = tokenizer.decode(pred[0].tolist())
        
        blue_score = sentence_bleu([target_formula], pred_formula)
        ed_score = edit_distance(pred_formula, target_formula) / len(pred_formula)
        em_score = exact_match(target_formula, pred_formula)
        em_score = len(em_score[0]) / (len(em_score[0]) + len(em_score[1]) + len(em_score[2]))
        vem_score = 1 if em_score == 1 else 0

        blue_scores.append(blue_score)
        ed_scores.append(ed_score)
        em_scores.append(em_score)
        vem_scores.append(vem_score)

        pbar.set_description(f"Processing Images, BLEU: {sum(blue_scores)/len(blue_scores):.5f}, Edit Distance: {sum(ed_scores)/len(ed_scores):.5f}, Exact Match: {sum(vem_scores)/len(vem_scores):.5f}")
        # pbar.set_description(f"Processing Images, BLEU: {sum(blue_scores)/len(blue_scores):.5f}, Edit Distance: {sum(ed_scores)/len(ed_scores):.5f}")

    # Print the final mean BLEU score
    print(f"Final Mean BLEU Score: {sum(blue_scores)/len(blue_scores):.5f}")
    print(f"Final Mean Edit Distance Score: {sum(ed_scores)/len(ed_scores):.5f}")
    print(f"Final Mean Old Exact Match Score: {sum(em_scores)/len(em_scores):.5f}")
    print(f"Final Mean Exact Match Score: {sum(vem_scores)/len(vem_scores):.5f}")

if __name__ == "__main__":
    main()