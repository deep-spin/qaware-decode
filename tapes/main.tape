task GetTrainingData
    < train_dir=@
    > train_src
    > train_tgt
    :: src_lang=@
    :: tgt_lang=@
{
    ln -s $train_dir/train.${src_lang} $train_src
    ln -s $train_dir/train.${tgt_lang} $train_tgt
}


task GetDevTestData
    < devtest_dir=@
    > dev_src
    > dev_tgt
    > test_src
    > test_tgt
    :: src_lang=@
    :: tgt_lang=@
{

    ln -s $devtest_dir/dev.${src_lang} $dev_src
    ln -s $devtest_dir/dev.${tgt_lang} $dev_tgt
    ln -s $devtest_dir/test.${src_lang} $test_src
    ln -s $devtest_dir/test.${tgt_lang} $test_tgt
}


task GetPretrained
    < pretrained_model=@
    > bpe_model
    > src_dict
    > tgt_dict
    :: src_lang=@
    :: tgt_lang=@
    :: is_multilingual=@
    :: bpe_type=@
{
    # check for the right bpe file
    if [ "$bpe_type" = "sentencepiece" ]; then
        ln -s $pretrained_model/sentencepiece.bpe.model $bpe_model
    elif [ "$bpe_type" = "fastbpe" ]; then
        ln -s $pretrained_model/bpecodes $bpe_model
    else
        echo "unknown or not supported bpe type"
        exit 1
    fi

    # multilingual models only come with one dict
    if [ "$is_multilingual" = true ]; then
        ln -s $pretrained_model/dict.txt $src_dict
        ln -s $pretrained_model/dict.txt $tgt_dict
    else
        ln -s $pretrained_model/dict.${src_lang}.txt $src_dict
        ln -s $pretrained_model/dict.${tgt_lang}.txt $tgt_dict
    fi
}


task TrainBPE
    < train_src=@GetTrainingData
    < train_tgt=@GetTrainingData
    > bpe_model
    > src_dict
    > tgt_dict
    :: .submitter=@ .mem=8000 .cpus=1
    :: repo=@
    :: bpe_type=@
    :: vocab_size=20000
{
    cat $train_src $train_tgt > train.all 
    python $repo/scripts/bpe_train.py train.all \
        --bpe-type $bpe_type \
        --model-prefix bpe_model \
        --vocab-file vocab \
        --vocab-size $vocab_size

    ln -s vocab $src_dict
    ln -s vocab $tgt_dict
    ln -s bpe_model.model $bpe_model
}


func ApplyBPE
    < raw_src
    < raw_tgt
    < bpe_model
    > prep_src
    > prep_tgt
    :: repo
    :: src_lang
    :: tgt_lang
    :: bpe_type
{
    python $repo/scripts/bpe_encode.py \
        --model $bpe_model --bpe-type $bpe_type --lang $src_lang \
            < $raw_src \
            > $prep_src
    python $repo/scripts/bpe_encode.py \
        --model $bpe_model --bpe-type $bpe_type --lang $tgt_lang \
            < $raw_tgt \
            > $prep_tgt
}


task ApplyBPETrain calls ApplyBPE
    < raw_src=$train_src@GetTrainingData
    < raw_tgt=$train_tgt@GetTrainingData
    < bpe_model=$bpe_model@TrainBPE
    > prep_src
    > prep_tgt
    :: bpe_type=@ src_lang=@ tgt_lang=@ repo=@


task ApplyBPEDev calls ApplyBPE
    < raw_src=$dev_src@GetDevTestData
    < raw_tgt=$dev_tgt@GetDevTestData
    < bpe_model=(
        UsePretrained:
            true=$bpe_model@GetPretrained
            false=$bpe_model@TrainBPE
        )
    > prep_src
    > prep_tgt
    :: bpe_type=@ src_lang=@ tgt_lang=@ repo=@


task ApplyBPETest calls ApplyBPE
    < raw_src=$test_src@GetDevTestData
    < raw_tgt=$test_tgt@GetDevTestData
    < bpe_model=(
        UsePretrained:
            true=$bpe_model@GetPretrained
            false=$bpe_model@TrainBPE
        )
    > prep_src
    > prep_tgt
    :: bpe_type=@ src_lang=@ tgt_lang=@ repo=@


task BinarizeData
    < train_src=(
        UsePretrained:
            true=/dev/null
            false=$prep_src@ApplyBPETrain
        )
    < train_tgt=(
        UsePretrained:
            true=/dev/null
            false=$prep_tgt@ApplyBPETrain
        )
    < dev_src=$prep_src@ApplyBPEDev
    < dev_tgt=$prep_tgt@ApplyBPEDev
    < test_src=$prep_src@ApplyBPETest
    < test_tgt=$prep_tgt@ApplyBPETest
    < src_dict=(
        UsePretrained:
            true=$src_dict@GetPretrained
            false=$src_dict@TrainBPE
        )
    < tgt_dict=(
        UsePretrained:
            true=$tgt_dict@GetPretrained
            false=$tgt_dict@TrainBPE
        )
    > bin_dir
    :: .submitter=@ .cpus=8 .mem=32000M .exclude=tir-1-11
    :: repo=@
    :: src_lang=@
    :: tgt_lang=@
    :: has_train=(
        UsePretrained:
            true=false
            false=true
        )
{
    # symlink for fairseq format
    ln -s $train_src train.${src_lang} 
    ln -s $train_tgt train.${tgt_lang} 
    ln -s $dev_src dev.${src_lang} 
    ln -s $dev_tgt dev.${tgt_lang} 
    ln -s $test_src test.${src_lang} 
    ln -s $test_tgt test.${tgt_lang} 

    fairseq-preprocess \
        --source-lang $src_lang --target-lang $tgt_lang \
        $([ "$has_train" = true ] && echo "--trainpref train" || echo "") \
        --validpref dev --testpref test \
        --thresholdsrc 0 --thresholdtgt 0 \
        --srcdict $src_dict --tgtdict $tgt_dict \
        --workers 8 \
        --destdir $bin_dir
}

task TrainModel
    < bin_dir=@BinarizeData
    < src_dict=@TrainBPE
    < tgt_dict=@TrainBPE
    < bpe_model=@TrainBPE
    > model_dir
    :: .submitter=@ .mem=16000 .gres="gpu:1" .cpus=2 
    :: src_lang=@
    :: tgt_lang=@
    :: bpe_type=@
    :: repo=@
    :: seed=@
{
    # TODO: parameters are currently hard-coded
    # later we should try to make variable while having a default
    lr=5e-4
    arch=transformer_iwslt_de_en
    patience=5
    max_epoch=1
    max_tokens=4096
    update_freq=8

    fairseq-train \
        $bin_dir \
        --fp16 \
        --task translation \
        --max-epoch $max_epoch \
        --log-interval 10 \
        --arch $arch --share-all-embeddings  \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 5. \
        --lr $lr --lr-scheduler inverse_sqrt  --warmup-updates 4000 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --dropout 0.3 --weight-decay 0.0001 \
        --max-tokens ${max_tokens} --update-freq ${update_freq} \
        --patience $patience \
        --eval-bleu \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        $([ "$bpe_type" = "fastbpe" ] && echo "--eval-bleu-detok moses" || echo "")  \
        --eval-bleu-remove-bpe $([ "$bpe_type" = "sentencepiece" ] && echo "sentencepiece" || echo "") \
        --eval-bleu-print-samples \
        --no-epoch-checkpoints \
        --save-dir $model_dir \
        --seed $seed

    # copy necessary files for a complete checkpoint
    if [ "$bpe_type" = "sentencepiece" ]; then
        ln -s $bpe_model $model_dir/sentencepiece.bpe.model 
    elif [ "$bpe_type" = "fastbpe" ]; then
        ln -s $bpe_model $model_dir/bpecodes 
    else
        echo "unknown or not supported bpe type"
        exit 1
    fi
    cp $src_dict  $model_dir/dict.${src_lang}.txt
    cp $tgt_dict  $model_dir/dict.${tgt_lang}.txt
    mv $model_dir/checkpoint_best.pt $model_dir/model.pt
}

func GenerateTranslations
    < bin_dir
    < model_dir
    > scores
    > predictions
    :: split
    :: repo
    :: src_lang
    :: tgt_lang
    :: comet_dir
    :: bpe_type
    :: is_multilingual
    :: infer_batchsize
    :: sampling
    :: sampling_topp
    :: nbest
{
    # TODO: test this
    multilingual_args=""
    if [ "$is_multilingual" = true ]; then
        multilingual_args+="--task translation_multi_simple_epoch "
        multilingual_args+="--decoder-langtok --encoder-langtok src "
        multilingual_args+="--lang-pairs $model_dir/language_pairs.txt "
        multilingual_args+="--fixed-dictionary $model_dir/dict.txt "
    fi

    fairseq-generate \
        $bin_dir \
        --fp16 \
        -s $src_lang -t $tgt_lang \
        --gen-subset $split \
        --batch-size $infer_batchsize \
        --path $model_dir/model.pt \
        $multilingual_args \
        --remove-bpe $([ "$bpe_type" = "sentencepiece" ] && echo "sentencepiece" || echo "") \
        $([ "$bpe_type" = "fastbpe" ] && echo "--tokenizer moses" || echo "")  \
        $([ "$sampling" = true ] && echo "--sampling" || echo "") \
        $([ ! -z "$sampling_topp"  ] && echo "--sampling-topp $sampling_topp" || echo "") \
        --beam $nbest \
        --nbest $nbest \
            > full_outputs 

        cat full_outputs | grep ^H | cut -c 3- | sort -n | cut -f2 > $scores
        cat full_outputs | grep ^H | cut -c 3- | sort -n | cut -f3- | \
            $([ "$bpe_type" = "fastbpe" ] && echo "sacremoses -l ${tgt_lang} detokenize" || echo "tee") > $predictions
}


task GenerateTranslationsDev calls GenerateTranslations
    < bin_dir=@BinarizeData
    < model_dir=(
        UsePretrained:
            true=$pretrained_model
            false=$model_dir@TrainModel
        )
    > scores
    > predictions
    :: .submitter=@ .gres="gpu:1" .cpus=5 .mem=32000M .time=0
    :: split=valid
    :: repo=@
    :: src_lang=@
    :: tgt_lang=@
    :: comet_dir=@
    :: bpe_type=@
    :: is_multilingual=@
    :: infer_batchsize=@
    :: sampling=@
    :: sampling_topp=@
    :: nbest=@


task GenerateTranslationsTest calls GenerateTranslations
    < bin_dir=@BinarizeData
    < model_dir=(
        UsePretrained:
            true=$pretrained_model
            false=$model_dir@TrainModel
        )
    > scores
    > predictions
    :: .submitter=@ .gres="gpu:1" .cpus=5 .mem=32000M .time=0
    :: split=test
    :: repo=@
    :: src_lang=@
    :: tgt_lang=@
    :: comet_dir=@
    :: bpe_type=@
    :: is_multilingual=@
    :: infer_batchsize=@
    :: sampling=@
    :: sampling_topp=@
    :: nbest=@


task RerankTranslations
    < dev_src=@GetDevTestData
    < dev_tgt=@GetDevTestData
    < dev_scores=$scores@GenerateTranslationsDev
    < dev_nbest=$predictions@GenerateTranslationsDev
    < test_src=@GetDevTestData
    < test_scores=$scores@GenerateTranslationsTest
    < test_nbest=$predictions@GenerateTranslationsTest
    > predictions
    :: .submitter=@ .gres="gpu:1" .cpus=5 .mem=32000M .time=0
    :: repo=@
    :: comet_dir=@
    :: rerank=@
    :: rerank_weights=@
    :: nbest=@
{
    if [ "$rerank" != false ]; then
        echo "$rerank_weights" > weights.txt
        
        if [ "$rerank" == "train" ]; then
            python $repo/scripts/rerank_prepare.py \
                $dev_nbest $dev_scores dev_features \
                --nbest $nbest \
                --add-cometsrc $comet_dir \
                --src $dev_src
            
            mv weights.txt weights_init.txt
            $repo/scripts/rerank_train.sh weights_init.txt dev_features $dev_tgt > weights.txt
        fi
        python $repo/scripts/rerank_prepare.py \
            $test_nbest $test_scores test_features \
            --nbest $nbest \
            --add-cometsrc $comet_dir \
            --src $test_src 

        $repo/scripts/rerank.sh weights.txt test_features > $predictions
    else
        awk "NR % $nbest == 1" $test_nbest > $predictions 
    fi
}


task ScoreTranslations
    < test_src=@GetDevTestData
    < test_tgt=@GetDevTestData
    < predictions=@RerankTranslations
    > score
    :: .submitter=@ .gres="gpu:1" .cpus=4 .mem=16000M
    :: repo=@
    :: src_lang=@
    :: tgt_lang=@
    :: comet_dir=@
{
    python $repo/scripts/score.py \
        $predictions $test_tgt --src $test_src \
        --comet $comet_dir > $score
}


summary TranslationQuality {
  of ScoreTranslations >  bleu comet {
    cat $score | grep -oP "COMET = \K[-0-9.]+" > $comet
    cat $score | grep -oP "BLEU = \K[-0-9.]+" > $bleu
  }
}