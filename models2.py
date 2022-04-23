import numpy as np
import spacy
from spacy.tokens import Doc
from spacy.training import Alignment

import torch
from transformers import AutoTokenizer, AutoModel
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents


COMBINATION_CHOICES = ['NONE', 'REDUCE_MEAN', 'REDUCE_MAX', 
                       'REDUCE_SUM', 'FIRST_TOKEN', 'LAST_TOKEN']


def combine_vectors(vectors, combination_type):
    """Combines a list vectors in to one vector.

    Args:
        vectors (List): the list of vectors to be combined.
        combination_type (str): the type of combination to be applied. See `COMBINATION_CHOICES`.
    """

    if hasattr(vectors, 'numpy'):
        vectors = vectors.numpy()
    if combination_type == 'REDUCE_MEAN':
        return np.average(vectors, axis=0)
    elif combination_type == 'REDUCE_MAX':
        return np.amax(vectors, axis=0)
    elif combination_type == 'REDUCE_SUM':
        return np.sum(vectors, axis=0)
    elif combination_type == 'FIRST_TOKEN':
        return vectors[0, :].copy()
    elif combination_type == 'LAST_TOKEN':
        return vectors[-1, :].copy()
    else:
        return vectors


def get_model(contextual=False, combination='REDUCE_MEAN', use_spacy=True, 
              transformer_model='roberta-base'):
    
    if use_spacy or contextual is False:
        if contextual:
            gpu = spacy.prefer_gpu()
            print('Using GPU:', gpu)
            spacy_model_name = 'en_core_web_trf'
        else:
            spacy_model_name = 'en_core_web_lg'

        # checks if model is downloaded
        if not spacy.util.is_package(spacy_model_name):
            spacy.cli.download(spacy_model_name)

        model = spacy.load(spacy_model_name)

        print('> Loaded model:', spacy_model_name)
        return model, None
        
    else:
        print(f'> Loading {transformer_model} tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained(transformer_model, add_special_tokens=True)
        print(f'> Loading {transformer_model} model...')
        model = model = AutoModel.from_pretrained(transformer_model, output_hidden_states = True)

        # Put the model in "evaluation" mode, meaning feed-forward operation.
        model.eval()
        return model, tokenizer


def is_tokenization_consistent(chunk, doc, tags):
        doc_tokens = [token.text for token in doc]
        
        if doc_tokens != chunk or len(doc_tokens) != len(tags):
            return False

        if hasattr(doc._, 'trf_data'):
            align_lengths = doc._.trf_data.align.lengths
            # check if at least one entry is zero
            if sum(align_lengths) < len(align_lengths):
                return False
        return True


def get_embeddings_spacy(model, chunk, tags):
    # Use predefined tokenization from dataset. For details, see
    # https://spacy.io/usage/linguistic-features#tokenization
    # An interesting discussion about tokenization trade-offs here: 
    # https://github.com/explosion/spaCy/issues/2011#issuecomment-367865510
    def custom_tokenizer(text):
        return Doc(model.vocab, words=chunk)
    model.tokenizer = custom_tokenizer
    chunk_str = ' '.join(chunk)
    doc = model(chunk_str)

    if not is_tokenization_consistent(chunk, doc, tags):
        print(f'Skipping sample: tokenization is inconsistent')
        # print('Document:', doc)
        return []

    embeddings = []
    for ii, token in enumerate(doc):
        if token.vector is None or np.isnan(token.vector).any() or np.isinf(token.vector.any()):
            print(f"Skipping token: found nan or inf in vector for token '{token}'")
            print('Document:', doc)
            # print('Chunk:', len(chunk), chunk)
            continue
        
        # print(token.vector.shape, doc.tensor.shape)
        embeddings.append(token.vector)

    return embeddings


def get_token_embeddings(embeddings, input_tokens, tokenized_text, combination_type):
    token_embeddings = []
    
    try:
        align = Alignment.from_strings(input_tokens, tokenized_text)
    except ValueError as err:
        # print(err)
        # print('tokenized_text:', tokenized_text)
        # print('input tokens:', input_tokens)
        return []
    
    token_start_idx = 1
    for length in align.x2y.lengths:
        if length == 0:
           return []

        token_end_idx = token_start_idx + length
        subword_embeddings = embeddings[token_start_idx:token_end_idx]
        token_embedding = combine_vectors(subword_embeddings, combination_type)
        token_embeddings.append(token_embedding)
        token_start_idx = token_end_idx

    return token_embeddings


def get_embeddings_spacy_contextual(model, input_tokens, tags, combination_type):

    # apply the same normalization as the original BERT tokenizer
    # more info here: https://huggingface.co/docs/tokenizers/python/latest/pipeline.html
    normalizer = normalizers.Sequence([NFD(), StripAccents()])
    tokens = [normalizer.normalize_str(token) for token in input_tokens]
    
    # Use predefined tokenization from dataset. For details, see
    # https://spacy.io/usage/linguistic-features#tokenization
    # An interesting discussion about tokenization trade-offs here: 
    # https://github.com/explosion/spaCy/issues/2011#issuecomment-367865510
    def custom_tokenizer(text):
        return Doc(model.vocab, words=tokens)
    model.tokenizer = custom_tokenizer
    text = ' '.join(tokens)
    doc = model(text)

    if not is_tokenization_consistent(tokens, doc, tags):
        print(f'Skipping sample: tokenization is inconsistent')
        # print('Document:', doc)
        return []

    tokenized_text = doc._.trf_data.tokens['input_texts'][0]
    tokenized_text = [token.replace('Ġ', '') for token in tokenized_text if token not in ['<s>', '</s>']]
    embeddings = doc._.trf_data.tensors[0][0]
    
    return get_token_embeddings(embeddings, tokens, tokenized_text, combination_type)


def get_embeddings_huggingface(model, tokenizer, input_tokens, combination_type, layer=-1):
    # apply the same normalization as the original BERT tokenizer
    # more info here: https://huggingface.co/docs/tokenizers/python/latest/pipeline.html
    normalizer = normalizers.Sequence([NFD(), StripAccents()])
    tokens = [normalizer.normalize_str(token) for token in input_tokens]

    text = ' '.join(tokens)
    
    # Tokenize our sentence with the BERT tokenizer.
    inputs = tokenizer.encode_plus(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs[2]
    
    # stack representations from all layers
    embeddings = torch.stack(hidden_states, dim=0)
    # remove batch dimension
    embeddings = torch.squeeze(embeddings, dim=1)
    # make the token dimension the first. Shape=(n_tokens, n_layers, embedding_dim)
    embeddings = embeddings.permute(1,0,2)
    # choose layer. Also could be a combination of layers.
    embeddings = embeddings[:,layer, :]
    
    tokenized_text = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0], skip_special_tokens=True)
    # tokenized_text = [token.replace('##', '') for token in tokenized_text]
    tokenized_text = [token.replace('Ġ', '') for token in tokenized_text]
    
    return get_token_embeddings(embeddings, tokens, tokenized_text, combination_type)
