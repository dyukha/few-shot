mkdir -p ../out/fewshot_128_knn_merge_mul_interpolation_0.4_few_contrastive
parallel -j4 --lb -n 3 'arr=(na 0 1 2 3); python fewshot_knn_interpolation.py {1} {2} {3} 128 merge_labels mul 0.4 few contrastive "[1, 2, 4, 8, 16]" ${arr[{%}]} |& tee ../out/fewshot_128_knn_merge_mul_interpolation_0.4_few_contrastive/{1}.txt' ::: \
    SST-2 sst-2 2 \
    cr cr 2 \
    QQP qqp/f1 2 \
    sst-5 sst-5 5 \
    mr mr 2 \
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

mkdir -p ../out/fewshot_128_knn_merge_mul_interpolation_0.6_few_contrastive
parallel -j4 --lb -n 3 'arr=(na 0 1 2 3); python fewshot_knn_interpolation.py {1} {2} {3} 128 merge_labels mul 0.6 few contrastive "[1, 2, 4, 8, 16]" ${arr[{%}]} |& tee ../out/fewshot_128_knn_merge_mul_interpolation_0.6_few_contrastive/{1}.txt' ::: \
    SST-2 sst-2 2 \
    cr cr 2 \
    QQP qqp/f1 2 \
    sst-5 sst-5 5 \
    mr mr 2 \
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

#mkdir -p ../out/fewshot_16_knn_merge_mul_interpolation
#parallel -j4 --lb -n 3 'arr=(na 0 1 2 3); python fewshot_knn_interpolation.py {1} {2} {3} 16 merge_labels mul drop_dense cross_entropy "[1, 2, 4, 8, 16]" ${arr[{%}]} |& tee ../out/fewshot_16_knn_merge_mul_interpolation/{1}.txt' ::: \
#    SST-2 sst-2 2 \
#    cr cr 2 \
#    mr mr 2 \
#    QQP qqp/f1 2 \
#    sst-5 sst-5 5 \
#    mpqa mpqa 2 \
#    subj subj 2 \
#    trec trec 6 \
#    CoLA cola 2 \
#    MNLI mnli 3 \
#    SNLI snli 3 \
#    QNLI qnli 2 \
#    RTE rte 2 \
#    MRPC mrpc/f1 2 \
#    STS-B sts-b/pearson 2 \

#mkdir -p ../out/fewshot_128_knn_merge_mul_linear_fix
#parallel -j4 --lb -n 3 'arr=(na 0 1 2 3); python fewshot_knn_linear_fix.py {1} {2} {3} 128 merge_labels mul "[1, 2, 4, 8, 16]" ${arr[{%}]} |& tee ../out/fewshot_128_knn_merge_mul_linear_fix/{1}.txt' ::: \
#    SST-2 sst-2 2 \
#    cr cr 2 \
#    mr mr 2 \
#    QQP qqp/f1 2 \
#    sst-5 sst-5 5 \
#    mpqa mpqa 2 \
#    subj subj 2 \
#    trec trec 6 \
#    CoLA cola 2 \
#    MNLI mnli 3 \
#    SNLI snli 3 \
#    QNLI qnli 2 \
#    RTE rte 2 \
#    MRPC mrpc/f1 2 \
#    STS-B sts-b/pearson 2 \
#
#mkdir -p ../out/fewshot_16_knn_merge_nomul
##parallel -j4 --lb -n 3 'arr=(na 0 1 2 3); python fewshot_knn.py {1} {2} {3} 128 merge_labels mul ${arr[{%}]} |& tee ../out/fewshot_128_knn_merge/{1}.txt' ::: \
#parallel -j4 --lb -n 3 'arr=(na 0 1 2 3); python fewshot_knn.py {1} {2} {3} 16 merge_labels nomul "[1, 3, 5, 7, 9]" ${arr[{%}]} |& tee ../out/fewshot_16_knn_merge_nomul/{1}.txt' ::: \
#    SST-2 sst-2 2 \
#    cr cr 2 \
#    mr mr 2 \
#    QQP qqp/f1 2 \
#    sst-5 sst-5 5 \
#    mpqa mpqa 2 \
#    subj subj 2 \
#    trec trec 6 \
#    CoLA cola 2 \
#    MNLI mnli 3 \
#    SNLI snli 3 \
#    QNLI qnli 2 \
#    RTE rte 2 \
#    MRPC mrpc/f1 2 \
#    STS-B sts-b/pearson 2 \
#
#mkdir -p ../out/fewshot_16_knn_merge_mul
#parallel -j4 --lb -n 3 'arr=(na 0 1 2 3); python fewshot_knn.py {1} {2} {3} 16 merge_labels mul "[1, 2, 4, 8, 16]" ${arr[{%}]} |& tee ../out/fewshot_16_knn_merge_mul/{1}.txt' ::: \
#    SST-2 sst-2 2 \
#    cr cr 2 \
#    mr mr 2 \
#    QQP qqp/f1 2 \
#    sst-5 sst-5 5 \
#    mpqa mpqa 2 \
#    subj subj 2 \
#    trec trec 6 \
#    CoLA cola 2 \
#    MNLI mnli 3 \
#    SNLI snli 3 \
#    QNLI qnli 2 \
#    RTE rte 2 \
#    MRPC mrpc/f1 2 \
#    STS-B sts-b/pearson 2 \
#
#mkdir -p ../out/fewshot_128_knn_merge_nomul
##parallel -j4 --lb -n 3 'arr=(na 0 1 2 3); python fewshot_knn.py {1} {2} {3} 128 merge_labels mul ${arr[{%}]} |& tee ../out/fewshot_128_knn_merge/{1}.txt' ::: \
#parallel -j4 --lb -n 3 'arr=(na 0 1 2 3); python fewshot_knn.py {1} {2} {3} 128 merge_labels nomul "[1, 3, 5, 7, 9]" ${arr[{%}]} |& tee ../out/fewshot_128_knn_merge_nomul/{1}.txt' ::: \
#    SST-2 sst-2 2 \
#    cr cr 2 \
#    mr mr 2 \
#    QQP qqp/f1 2 \
#    sst-5 sst-5 5 \
#    mpqa mpqa 2 \
#    subj subj 2 \
#    trec trec 6 \
#    CoLA cola 2 \
#    MNLI mnli 3 \
#    SNLI snli 3 \
#    QNLI qnli 2 \
#    RTE rte 2 \
#    MRPC mrpc/f1 2 \
#    STS-B sts-b/pearson 2 \
