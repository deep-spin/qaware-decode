import argparse
from collections import Counter

import sacrebleu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hyp")
    parser.add_argument("ref")

    args = parser.parse_args()

    with open(args.hyp, encoding='utf-8') as hyp_f:
        hyps = [line.strip() for line in hyp_f.readlines()]
    with open(args.ref, encoding='utf-8') as ref_f:
        refs = [line.strip() for line in ref_f.readlines()]

    print(sacrebleu.corpus_bleu(hyps, [refs]).format())

if __name__ == "__main__":
    main()