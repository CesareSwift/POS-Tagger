
import argparse
import pyconll
import json


def conllu2json(source_path, output_path):
    """Extract POS tags from a file in CoNLL-U format and convert to JSON lines.

    Args:
        source_path (string): path to source file in CoNLL-U format
        output_path (string): path to destination file in JSON lines format
    """
    dataset = pyconll.load_from_file(source_path)
    sentences = []

    for sentence in dataset:
        item = {
            'text': sentence.text,
            'tokens': [],
            'tags': []
        }
        
        for token in sentence:
            if token.upos:
                item['tokens'].append(token.form)
                item['tags'].append(token.upos)

        sentences.append(item)

    with open(output_path, "w") as fh:
        for sentence in sentences:
            fh.write(json.dumps(sentence))
            fh.write("\n")

        print(f'Saved {len(sentences)} sentences to {output_path}')


def setup_argparse():
    p = argparse.ArgumentParser()
    p.add_argument('--source_path', help='path to dataset in CoNLL-U format')
    p.add_argument('--output_path', default='output.json', help='path of output file in JSON format')
    return p.parse_args()


if __name__ == "__main__":
    args = setup_argparse()
    conllu2json(args.source_path, args.output_path)

# Usage
# python utils/conllu2json.py --source_path data/en_ewt-ud-train.conllu --output_path data/en_ewt-ud-train.json