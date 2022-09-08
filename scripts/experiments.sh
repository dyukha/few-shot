(bash run_pf.sh sst-5 sst-5 0 |& tee out_pf_sst-5.txt) &
(bash run_pf.sh SST-2 sst-2 1 |& tee out_pf_sst-2.txt) &
(bash run_pf_filter.sh sst-5 sst-5 2 |& tee out_pf_sst-5_filter.txt) &
(bash run_pf_filter.sh SST-2 sst-2 3 |& tee out_pf_sst-2_filter.txt) &
wait
(bash run_zeroshot.sh sst-5 sst-5 0 |& tee out_zero_sst-5.txt) &
(bash run_zeroshot.sh SST-2 sst-2 1 |& tee out_zero_sst-2.txt) &
wait
(bash run_fewshot.sh sst-5 sst-5 0 |& tee out_few_sst-5.txt) &
(bash run_fewshot.sh SST-2 sst-2 1 |& tee out_few_sst-2.txt) &
(bash run_fewshot_filter.sh sst-5 sst-5 2 |& tee out_few_sst-5_filter.txt) &
(bash run_fewshot_filter.sh SST-2 sst-2 3 |& tee out_few_sst-2_filter.txt) &
wait
