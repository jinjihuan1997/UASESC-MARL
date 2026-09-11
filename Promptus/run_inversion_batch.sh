#!/bin/bash

# 通用设置
FRAME_PATH="data/sky"
MAX_ID=37

echo "Starting batch inversion process..."

# Rank 2
python inversion_all.py -frame_path $FRAME_PATH -max_id $MAX_ID -rank 2 -interval 5
python inversion_all.py -frame_path $FRAME_PATH -max_id $MAX_ID -rank 2 -interval 6
python inversion_all.py -frame_path $FRAME_PATH -max_id $MAX_ID -rank 2 -interval 7
python inversion_all.py -frame_path $FRAME_PATH -max_id $MAX_ID -rank 2 -interval 8
python inversion_all.py -frame_path $FRAME_PATH -max_id $MAX_ID -rank 2 -interval 9
python inversion_all.py -frame_path $FRAME_PATH -max_id $MAX_ID -rank 2 -interval 10
python inversion_all.py -frame_path $FRAME_PATH -max_id $MAX_ID -rank 2 -interval 11
python inversion_all.py -frame_path $FRAME_PATH -max_id $MAX_ID -rank 2 -interval 12
python inversion_all.py -frame_path $FRAME_PATH -max_id $MAX_ID -rank 2 -interval 13
python inversion_all.py -frame_path $FRAME_PATH -max_id $MAX_ID -rank 2 -interval 14
python inversion_all.py -frame_path $FRAME_PATH -max_id $MAX_ID -rank 2 -interval 15
python inversion_all.py -frame_path $FRAME_PATH -max_id $MAX_ID -rank 2 -interval 25

# Rank 4
python inversion_all.py -frame_path $FRAME_PATH -max_id $MAX_ID -rank 4 -interval 2
python inversion_all.py -frame_path $FRAME_PATH -max_id $MAX_ID -rank 4 -interval 5
python inversion_all.py -frame_path $FRAME_PATH -max_id $MAX_ID -rank 4 -interval 10
python inversion_all.py -frame_path $FRAME_PATH -max_id $MAX_ID -rank 4 -interval 15

# Rank 8
python inversion_all.py -frame_path $FRAME_PATH -max_id $MAX_ID -rank 8 -interval 10
python inversion_all.py -frame_path $FRAME_PATH -max_id $MAX_ID -rank 8 -interval 15
python inversion_all.py -frame_path $FRAME_PATH -max_id $MAX_ID -rank 8 -interval 25

# Rank 12
python inversion_all.py -frame_path $FRAME_PATH -max_id $MAX_ID -rank 12 -interval 2
python inversion_all.py -frame_path $FRAME_PATH -max_id $MAX_ID -rank 12 -interval 5
python inversion_all.py -frame_path $FRAME_PATH -max_id $MAX_ID -rank 12 -interval 10
python inversion_all.py -frame_path $FRAME_PATH -max_id $MAX_ID -rank 12 -interval 15
python inversion_all.py -frame_path $FRAME_PATH -max_id $MAX_ID -rank 12 -interval 25

# Rank 16
python inversion_all.py -frame_path $FRAME_PATH -max_id $MAX_ID -rank 16 -interval 2
python inversion_all.py -frame_path $FRAME_PATH -max_id $MAX_ID -rank 16 -interval 5
python inversion_all.py -frame_path $FRAME_PATH -max_id $MAX_ID -rank 16 -interval 10
python inversion_all.py -frame_path $FRAME_PATH -max_id $MAX_ID -rank 16 -interval 15
python inversion_all.py -frame_path $FRAME_PATH -max_id $MAX_ID -rank 16 -interval 25

# Rank 24
python inversion_all.py -frame_path $FRAME_PATH -max_id $MAX_ID -rank 24 -interval 5
python inversion_all.py -frame_path $FRAME_PATH -max_id $MAX_ID -rank 24 -interval 10
python inversion_all.py -frame_path $FRAME_PATH -max_id $MAX_ID -rank 24 -interval 25

# Rank 60 (图片最下方的一个)
python inversion_all.py -frame_path $FRAME_PATH -max_id $MAX_ID -rank 60 -interval 25

echo "All inversion tasks completed."