mkdir -p ../out/fewshot_coresets_128_merge
parallel -j4 --lb -n 3 'arr=(na 0 1 2 3); python fewshot_coresets.py {1} {2} {3} 128 merge_labels mul ${arr[{%}]} |& tee ../out/fewshot_coresets_128_merge/{1}.txt' ::: \
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
