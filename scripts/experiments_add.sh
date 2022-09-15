mkdir -p ../out/fewshot_similar_128_separate
#parallel -j4 --lb -n 3 'echo {1} {2} {4} {6}' ::: \
#    1 2 4 8 ::: \
#    trec trec 6 \

#parallel -j4 --lb -n 3 'arr=(na 0 1 2 3); python fewshot_similar.py {2} {4} {6} 128 separate_labels mul "[{1}]" ${arr[{%}]} |& tee ../out/fewshot_similar_128_separate/{2}.txt' ::: \
#    2 4 8 16 ::: \
#    QQP qqp/f1 2 \

parallel -j4 --lb -n 3 'arr=(na 0 1 2 3); python fewshot_similar.py {1} {2} {3} 128 separate_labels mul "[1, 2, 4, 8, 16]" ${arr[{%}]} |& tee ../out/fewshot_similar_128_separate/{1}.txt' ::: \
    QQP qqp/f1 2 \
