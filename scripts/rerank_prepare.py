import argparse

import torch
import sentencepiece as spm

from comet import download_model, load_from_checkpoint
from transquest.algo.sentence_level.monotransquest.run_model import MonoTransQuestModel

COMETSRC_MODEL = "wmt20-comet-qe-da"
TRANSQUEST_Model = "TransQuest/monotransquest-da-multilingual"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("hyps")
    parser.add_argument("scores")
    parser.add_argument("formatted")
    parser.add_argument("--spm", default=False)
    parser.add_argument("--nbest", required=True, type=int)
    parser.add_argument("--add-cometsrc", default=None)
    parser.add_argument("--add-transquest", default=None)
    parser.add_argument("--comet-path", default=None)
    parser.add_argument("--src", default=None)
    args = parser.parse_args()

    with open(args.hyps, encoding='utf-8') as hyp_f:
        hyps = [line.strip() for line in hyp_f.readlines()]

    with open(args.scores, encoding='utf-8') as score_f:
        scores = [float(line.strip()) for line in score_f.readlines()]

    def src_hyp_iterator(srcs, hyps):
        assert len(srcs) * args.nbest == len(hyps), f"{len(srcs) * args.nbest} != {len(hyps)}"
        for i, src in enumerate(srcs):
            for j in range(args.nbest):
                hyp = hyps[i*args.nbest + j]
                yield src, hyp

    if args.add_cometsrc is not None:
        assert args.src is not None, "source needs to be provided to use COMET"
        with open(args.src, encoding='utf-8') as src_f:
            srcs = [line.strip() for line in src_f.readlines()]

        # download comet and load
        comet_path = download_model(COMETSRC_MODEL, args.add_cometsrc)
        comet_model = load_from_checkpoint(comet_path)
        comet_input = [
            {"src": src, "mt": mt} for src, mt in src_hyp_iterator(srcs, hyps)
        ]
        comet_scores, _ = comet_model.predict(
            comet_input, num_workers=4
        )

    if args.add_transquest is not None:
        assert args.src is not None, "source needs to be provided to use COMET"
        with open(args.src, encoding='utf-8') as src_f:
            srcs = [line.strip() for line in src_f.readlines()]

        transquest_model = MonoTransQuestModel(
            "xlmroberta", "TransQuest/monotransquest-da-multilingual", 
            num_labels=1, 
            use_cuda=torch.cuda.is_available()
        )
        transquest_input = [[src, mt] for src, mt in src_hyp_iterator(srcs, hyps)]
        transquest_scores, _ = model.predict(transquest_input)

    with open(args.formatted, "w", encoding='utf-8') as formatted_f:
        for i, (hyp, score) in enumerate(zip(hyps, scores)):
            sample = i // args.nbest
            parts = [str(sample), hyp, f"{score}"]
            features = [f"logprob={score}"]
            
            if args.add_cometsrc is not None:
                features.append(f"cometsrc={comet_scores[i]}")
            
            if args.add_transquest is not None:
                features.append(f"transquest={transquest_scores[i]}")
            

            parts.append(" ".join(features))
            print(" ||| ".join(parts), file=formatted_f)
        
if __name__ == "__main__":
    main()
