#1/usr/bin/bash
CANDIDATE_PATH=/home/lvloi/projects/vlsp-2022/submission-results
python=/home/lvloi/projects/vlsp-2022/seq2seq/.venv/bin/python
GOLDPATH=/home/lvloi/projects/vlsp-2022/data/vlsp-2022/gold_valid.out
OUTPUTPATH=/home/lvloi/projects/vlsp-2022/rouge.txt

for line in $(ls -l $CANDIDATE_PATH | awk '{print $10}'); do
if [[ $line =~ valid ]]; then
echo $line >> $OUTPUTPATH
$python cal_rouge.py --candidate-path $CANDIDATE_PATH/$line --reference-path $GOLDPATH >> $OUTPUTPATH 2>&1
fi
done