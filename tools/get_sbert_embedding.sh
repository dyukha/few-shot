for K in 32 64 128 256; do
  python tools/generate_k_shot_data.py --k $K &
done
wait

MODEL=roberta-large

for K in 32 64 128 256; do
  ##python tools/get_sbert_embedding.py --sbert_model $MODEL --task SST-2 sst-5 mr cr mpqa subj trec CoLA MRPC QQP STS-B MNLI SNLI QNLI RTE
  CUDA_VISIBLE_DEVICES=0 python tools/get_sbert_embedding.py --sbert_model $MODEL --k $K --task SST-2 sst-5 mr STS-B --seed 13 21 42 87 100 &
  CUDA_VISIBLE_DEVICES=1 python tools/get_sbert_embedding.py --sbert_model $MODEL --k $K --task cr mpqa subj MNLI --seed 13 21 42 87 100 &
  CUDA_VISIBLE_DEVICES=2 python tools/get_sbert_embedding.py --sbert_model $MODEL --k $K --task trec CoLA MRPC QQP --seed 13 21 42 87 100 &
  CUDA_VISIBLE_DEVICES=3 python tools/get_sbert_embedding.py --sbert_model $MODEL --k $K --task SNLI QNLI RTE --seed 13 21 42 87 100 &
  wait
done
#echo "finished generating"
#
##python tools/get_sbert_embedding.py --sbert_model $MODEL --seed 42 --do_test --task SST-2 sst-5 mr cr mpqa subj trec CoLA MRPC QQP STS-B MNLI SNLI QNLI RTE
#CUDA_VISIBLE_DEVICES=0 python tools/get_sbert_embedding.py --sbert_model $MODEL --seed 42 --do_test --task SST-2 sst-5 mr STS-B &
#CUDA_VISIBLE_DEVICES=1 python tools/get_sbert_embedding.py --sbert_model $MODEL --seed 42 --do_test --task cr mpqa subj MNLI &
#CUDA_VISIBLE_DEVICES=2 python tools/get_sbert_embedding.py --sbert_model $MODEL --seed 42 --do_test --task trec CoLA MRPC QQP &
#CUDA_VISIBLE_DEVICES=3 python tools/get_sbert_embedding.py --sbert_model $MODEL --seed 42 --do_test --task SNLI QNLI RTE &
#wait
#echo "finished generating for the test split for seed 42"


for K in 32 64 128 256; do
  for seed in 13 21 42 87 100; do
    for task in SST-2 sst-5 mr cr mpqa subj trec CoLA MRPC QQP STS-B SNLI QNLI RTE; do
      cp data/k-shot/$task/16-42/test_sbert-$MODEL.npy data/k-shot/$task/$K-$seed/
    done

    cp data/k-shot/MNLI/16-42/test_matched_sbert-$MODEL.npy data/k-shot/MNLI/$K-$seed/
    cp data/k-shot/MNLI/16-42/test_mismatched_sbert-$MODEL.npy data/k-shot/MNLI/$K-$seed/
  done
done
