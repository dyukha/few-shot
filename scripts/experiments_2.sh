#bash run_pf_new.sh SST-2 sst-2 |& tee out_pf_sst-2.txt
#bash run_pfd_filter_new.sh SST-2 sst-2 |& tee out_pfd_filter_sst-2.txt
#bash run_pf_new.sh cr cr |& tee out_pf_cr.txt
#bash run_pfd_filter_new.sh cr cr |& tee out_pfd_filter_cr.txt
#bash run_pf_new.sh mr mr |& tee out_pf_mr.txt
#bash run_pfd_filter_new.sh mr mr |& tee out_pfd_filter_mr.txt
#bash run_pf_new.sh sst-5 sst-5 |& tee out_pf_sst-5.txt
#bash run_pfd_filter_new.sh sst-5 sst-5 |& tee out_pfd_filter_sst-5.txt

for i in "SST-2 sst-2" "cr cr" "mr mr" "sst-5 sst-5" "mpqa mpqa" "subj subj" "trec trec" "CoLA cola" "MNLI mnli" "SNLI snli" \
         "QNLI qnli" "RTE rte" "MRPC mrpc" "STS-B sts-b" "QQP qqp"
do
  pair=( $i )
  bash run_pfd_filter_new.sh ${pair[0]} ${pair[1]} |& tee ../out/pfd_filter_${pair[0]}.txt
done

#for i in "QQP qqp/f1" "MRPC mrpc/f1" "STS-B sts-b/pearson"
