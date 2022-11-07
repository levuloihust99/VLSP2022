#!/usr/bin/bash
python=/home/lvloi/projects/vlsp-2022/seq2seq/.venv/bin/python
OUTPUTPATH=/home/lvloi/projects/vlsp-2022/submission-results
DATAPATH=/home/lvloi/projects/vlsp-2022/data/vlsp-2022/jsonl
valid_file=$DATAPATH/vlsp_2022_abmusu_validation_data_new.jsonl
test_file=$DATAPATH/vlsp_abmusu_test_data.jsonl
checkpoints_file=/home/lvloi/projects/vlsp-2022/scripts/checkpoints.txt

cd /home/lvloi/projects/vlsp-2022/seq2seq

while read line; do
model_path=/home/lvloi/projects/vlsp-2022/submission-checkpoints/$line
[[ $line =~ [0-9]+ ]]
for num_beams in 1 5; do
$python -m nn.predict --test-file $valid_file \
    --model-path $model_path \
    --model-type base \
    --gpuid 1 \
    --mode concat \
    --max-length 500 \
    --num-beams $num_beams \
    --write-textline True \
    --output-path $OUTPUTPATH/candidate_valid_${BASH_REMATCH[0]}_beam$num_beams.out
$python -m nn.predict --test-file $test_file \
    --model-path $model_path \
    --model-type base \
    --gpuid 1 \
    --mode concat \
    --max-length 500 \
    --num-beams $num_beams \
    --write-textline True \
    --output-path $OUTPUTPATH/candidate_test_${BASH_REMATCH[0]}_beam$num_beams.out
done
done < $checkpoints_file