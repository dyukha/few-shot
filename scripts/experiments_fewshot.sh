mkdir -p ../out/fewshot
parallel -j4 --lb -n 3 'arr=(na 0 1 2 3); bash fewshot.sh {1} {2} {3} ${arr[{%}]} |& tee ../out/fewshot/{1}.txt' ::: \
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
#    SST-2 sst-2 2 \
#    cr cr 2 \
#    mr mr 2 \

#
#(bash infer.sh SST-2 sst-2 42 saved_models/SST-2-42 0 |& tee $dir/infer_sst-2.txt) &
#(bash infer.sh sst-5 sst-5 42 saved_models/sst-5-42 1 |& tee $dir/infer_sst-5.txt) &
#(bash infer.sh cr cr 42 saved_models/cr-42 2 |& tee $dir/infer_cr.txt) &
#(bash infer.sh mr mr 42 saved_models/mr-42 3 |& tee $dir/infer_mr.txt) &
#wait