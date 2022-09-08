# Important: comment the last line in ../run_experiment.sh

mkdir -p ../out
parallel -j4 --lb -n 4 'bash run_pfd_filter_fixed_params.sh {1} {2} 42 {3} {4} $(expr {%} - 1) |& tee ../out/fixed_{1}.txt' ::: \
    mpqa mpqa 2 1e-5 \
    QQP qqp/f1 4 2e-5 \
    subj subj 4 1e-5 \
    trec trec 8 1e-5 \
    CoLA cola 2 1e-5 \
    MNLI mnli 4 1e-5 \
    SNLI snli 8 1e-5 \
    QNLI qnli 8 1e-5 \
    RTE rte 2 1e-5 \
    MRPC mrpc/f1 4 1e-5 \
    STS-B sts-b/pearson 4 2e-5 \

#    SST-2 sst-2 4 2e-5 \
#    sst-5 sst-5 4 2e-5 \
#    cr cr 4 2e-5 \
#    mr mr 4 2e-5 \

#(bash run_pfd_filter_fixed_params.sh SST-2 sst-2 42 4 2e-5 0 |& tee $dir/fixed_sst-2.txt) &
#(bash run_pfd_filter_fixed_params.sh sst-5 sst-5 42 4 2e-5 1 |& tee $dir/fixed_sst-5.txt) &
#(bash run_pfd_filter_fixed_params.sh cr cr 42 4 2e-5 2 |& tee $dir/fixed_cr.txt) &
#(bash run_pfd_filter_fixed_params.sh mr mr 42 4 2e-5 3 |& tee $dir/fixed_mr.txt) &
#wait
#(bash run_pfd_filter_fixed_params.sh SST-2 sst-2 42 4 2e-5 0 |& tee $dir/fixed_sst-2.txt) &
#(bash run_pfd_filter_fixed_params.sh sst-5 sst-5 42 4 2e-5 1 |& tee $dir/fixed_sst-5.txt) &
#(bash run_pfd_filter_fixed_params.sh cr cr 42 4 2e-5 2 |& tee $dir/fixed_cr.txt) &
#(bash run_pfd_filter_fixed_params.sh mr mr 42 4 2e-5 3 |& tee $dir/fixed_mr.txt) &
#wait