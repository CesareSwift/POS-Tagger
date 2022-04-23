import pickle
import argparse

import numpy as np
from tqdm import tqdm

from utils.helper import read_jsonl
from models import get_embeddings_spacy_contextual, get_model
from models import get_embeddings_huggingface, get_embeddings_spacy
from models import COMBINATION_CHOICES
        

def get_dataset_embeddings(source_path, model, max_tokens=145, tokenizer=None, combination='REDUCE_MEAN'):

    def fix_symbols(text):
        text = text.replace('”', '"').replace('“', '"')
        text = text.replace('’', "'").replace('‘', "'")
        # '-' seems repeated here, but they are different characters
        text = text.replace('—', '-').replace('–', '-')
        text = text.replace('…', '...')
        text = text.replace('♥', ' ')
        return text
     
    def chunks(tokens, n, tags=None):
        if tags:
            for i in range(0, len(tokens), n):
                yield tokens[i:i+n], tags[i:i+n]
        else:
            for i in range(0, len(tokens), n):
                yield tokens[i:i+n], None

    embeddings, labels = [], []

    print('> Extracting embeddings for dataset', source_path)
    
    for sample in tqdm(read_jsonl(source_path)):
        gold_tokens = sample['tokens']
        gold_tags = sample['tags']

        # BERT max is 512 wordpiece tokens at once, and there is one sample that exceeeds it
        if len(gold_tokens) > max_tokens:
            text_chunks, tag_chunks = chunks(gold_tokens, max_tokens, tags=gold_tags)
        else:
            text_chunks, tag_chunks = [gold_tokens], [gold_tags]

        for chunk, tags in zip(text_chunks, tag_chunks):

            chunk = [fix_symbols(tok) for tok in chunk]
            chunk = [tok for tok in chunk if len(tok.strip()) > 0]
            
            if tokenizer:
                sample_embeds = get_embeddings_huggingface(model, tokenizer, chunk, combination)
            elif combination:
                sample_embeds = get_embeddings_spacy_contextual(model, chunk, tags, combination)
            else:
                sample_embeds = get_embeddings_spacy(model, chunk, tags)

            if len(sample_embeds) != len(tags):
                print(f'Skipping sample: embedding len ({len(sample_embeds)}) does not match number of tags ({len(tags)})')
                continue
            embeddings.extend(sample_embeds)
            labels.extend(tags)
            
        
    # save processed split of corpus, with matrix of number_samples x features, list of labels
    corpus = [np.vstack(embeddings), labels]

    # print number of entities found in each section for information
    print("> Processed {} tokens".format(len(corpus[0])))
    print('> Embeddings shape:', corpus[0].shape)
    return corpus


def save_embeddings(target_path=None, contextual=False, combination='REDUCE_MEAN'):
    model, tokenizer = get_model(contextual=contextual, combination=combination, use_spacy=False)
    
    source_paths = [
        ('TRAINING', 'data/en_ewt-ud-train.json'),
        ('VALIDATION', 'data/en_ewt-ud-dev.json'),
        ('TESTING', 'data/en_ewt-ud-test.json')
    ]

    corpus = {}
    if not contextual:
        combination = None

    for path in source_paths:
        corpus[path[0]] = get_dataset_embeddings(path[1], model, tokenizer=tokenizer,
                                                 combination=combination)

    if target_path is None:
        target_path = 'data/en_ewt-ud-embeds'
        if contextual:
            target_path += f'-{combination.lower()}'
        target_path += '.pkl'

    with open(target_path, "wb") as fout:
        pickle.dump(corpus, fout)

    print(f"Saved full processed corpus to {target_path}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--target_path', help='path to save embeddings file')
    p.add_argument('--contextual', action='store_true', help='use contextual embeddings from a Transformer model')
    p.add_argument('--combination', choices=COMBINATION_CHOICES, default='REDUCE_MEAN', required=False)
    args = p.parse_args()

    save_embeddings(args.target_path, contextual=args.contextual, combination=args.combination)


# Usage
# python embedding.py --contextual --combination REDUCE_MEAN
    