#!/usr/bin/bash
submission_cp_dir=/home/lvloi/projects/vlsp-2022/submission-checkpoints
checkpoint_state_file=/home/lvloi/projects/vlsp-2022/bash/checkpoints.txt
while read line; do
echo "Downloading $line ..."
[[ $line =~ [0-9]+ ]]
scp "cist-P100:/media/lvloi/backup/A100/vlsp-2022/seq2seq/tmp-vimds+abmusu/pytorch_model_${BASH_REMATCH[0]}.bin" $submission_cp_dir
done < $checkpoint_state_file