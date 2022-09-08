dir=../out
mkdir -p $dir
bash run_few_new.sh SST-2 sst-2 8 |& tee $dir/few_sst-2_8.txt
bash run_few_new.sh cr cr 8 |& tee $dir/few_cr_8.txt
bash run_few_new.sh mr mr 8 |& tee $dir/few_mr_8.txt
bash run_few_new.sh sst-5 sst-5 8 |& tee $dir/few_sst-5_8.txt
bash run_few_new.sh SST-2 sst-2 4 |& tee $dir/few_sst-2_4.txt
bash run_few_new.sh cr cr 4 |& tee $dir/few_cr_4.txt
bash run_few_new.sh mr mr 4 |& tee $dir/few_mr_4.txt
bash run_few_new.sh sst-5 sst-5 4 |& tee $dir/few_sst-5_4.txt
bash run_few_new.sh SST-2 sst-2 2 |& tee $dir/few_sst-2_2.txt
bash run_few_new.sh cr cr 2 |& tee $dir/few_cr_2.txt
bash run_few_new.sh mr mr 2 |& tee $dir/few_mr_2.txt
bash run_few_new.sh sst-5 sst-5 2 |& tee $dir/few_sst-5_2.txt
bash run_few_new.sh SST-2 sst-2 1 |& tee $dir/few_sst-2_1.txt
bash run_few_new.sh cr cr 1 |& tee $dir/few_cr_1.txt
bash run_few_new.sh mr mr 1 |& tee $dir/few_mr_1.txt
bash run_few_new.sh sst-5 sst-5 1 |& tee $dir/few_sst-5_1.txt
bash run_few_new.sh SST-2 sst-2 16 |& tee $dir/few_sst-2_16.txt
bash run_few_new.sh cr cr 16 |& tee $dir/few_cr_16.txt
bash run_few_new.sh mr mr 16 |& tee $dir/few_mr_16.txt
bash run_few_new.sh sst-5 sst-5 16 |& tee $dir/few_sst-5_16.txt

#MODEL=roberta-large
#for K in 1 2 4 8 16
#do
#  ##python tools/get_sbert_embedding.py --sbert_model $MODEL --task SST-2 sst-5 mr cr mpqa subj trec CoLA MRPC QQP STS-B MNLI SNLI QNLI RTE
#  (cd .. ; CUDA_VISIBLE_DEVICES=0 python tools/get_sbert_embedding.py --sbert_model $MODEL --k $K --task SST-2 sst-5 mr STS-B --seed 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99) &
#  (cd .. ; CUDA_VISIBLE_DEVICES=1 python tools/get_sbert_embedding.py --sbert_model $MODEL --k $K --task cr mpqa subj MNLI --seed 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99) &
#  (cd .. ; CUDA_VISIBLE_DEVICES=2 python tools/get_sbert_embedding.py --sbert_model $MODEL --k $K --task trec CoLA MRPC QQP --seed 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99) &
#  (cd .. ; CUDA_VISIBLE_DEVICES=3 python tools/get_sbert_embedding.py --sbert_model $MODEL --k $K --task SNLI QNLI RTE --seed 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99) &
#  wait
#done
