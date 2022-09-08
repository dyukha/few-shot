mkdir -p ../out/infer
parallel -j4 --lb -n 2 'arr=(na 0 1 2 3); bash infer.sh {1} {2} 42 saved_models/{1}-42 ${arr[{%}]} |& tee ../out/infer/{1}-42.txt' ::: \
    SST-2 sst-2 \
    sst-5 sst-5 \
    cr cr \
    mr mr \
    mpqa mpqa \
    QQP qqp/f1 \
    subj subj \
    trec trec \
    CoLA cola \
    MNLI mnli \
    SNLI snli \
    QNLI qnli \
    RTE rte \
    MRPC mrpc/f1 \
    STS-B sts-b/pearson \

#
#(bash infer.sh SST-2 sst-2 42 saved_models/SST-2-42 0 |& tee $dir/infer_sst-2.txt) &
#(bash infer.sh sst-5 sst-5 42 saved_models/sst-5-42 1 |& tee $dir/infer_sst-5.txt) &
#(bash infer.sh cr cr 42 saved_models/cr-42 2 |& tee $dir/infer_cr.txt) &
#(bash infer.sh mr mr 42 saved_models/mr-42 3 |& tee $dir/infer_mr.txt) &
#wait