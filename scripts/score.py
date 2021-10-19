import argparse
from collections import Counter

from comet import download_model, load_from_checkpoint

import sacrebleu


COMET_MODEL="wmt20-comet-da"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hyp", type=str)
    parser.add_argument("ref", type=str)
    parser.add_argument("--comet", type=str)
    parser.add_argument("--src", type=str)

    args = parser.parse_args()

    with open(args.hyp, encoding='utf-8') as hyp_f:
        hyps = [line.strip() for line in hyp_f.readlines()]
    with open(args.ref, encoding='utf-8') as ref_f:
        refs = [line.strip() for line in ref_f.readlines()]

    print(sacrebleu.corpus_bleu(hyps, [refs]).format())
    if args.comet is not None:
        assert args.src is not None, "source needs to be provided to use COMET"
        with open(args.src) as src_f:
            srcs = [line.strip() for line in src_f.readlines()]

        # download comet and load
        comet_path = download_model(COMET_MODEL, args.comet)
        comet_model = load_from_checkpoint(comet_path)

        print("running comet evaluation....")
        comet_input = [
            {"src": src, "mt": mt, "ref": ref} for src, mt, ref in zip(srcs, hyps, refs)
        ]
        _, comet_score = comet_model.predict(comet_input, num_workers=4)
        print(f"COMET = {comet_score:.4f}")

if __name__ == "__main__":
    main()