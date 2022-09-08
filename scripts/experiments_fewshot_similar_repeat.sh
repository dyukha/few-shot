inputs='[(1, 1), (1, 2), (1, 4), (1, 8), (1, 16), (2, 1), (2, 2), (2, 4), (2, 8), (4, 1), (4, 2), (4, 4), (8, 1), (8, 2), (16, 1)]'

mkdir -p ../out/fewshot_similar_repeat_16_separate
export inputs
parallel -j4 --lb -n 3 'arr=(na 0 1 2 3); python fewshot_similar_repeat.py {1} {2} {3} 16 separate_labels mul "$inputs" ${arr[{%}]} |& tee ../out/fewshot_similar_repeat_16_separate/{1}.txt' ::: \
    SST-2 sst-2 2 \
    cr cr 2 \
    mr mr 2 \
    QQP qqp/f1 2 \
    sst-5 sst-5 5 \
    mpqa mpqa 2 \
    subj subj 2 \
    trec trec 6 \
    CoLA cola 2 \
    MNLI mnli 3 \
    SNLI snli 3 \
    QNLI qnli 2 \
    RTE rte 2 \
    MRPC mrpc/f1 2 \
    STS-B sts-b/pearson 2 \
