VIS_SAVE_PATH="results/VideoLISA-RefDAVIS"
OUTPUT_DIR=$VIS_SAVE_PATH"_processed"


python evaluation/refytvos/post_process_davis.py --src_dir $VIS_SAVE_PATH


ANNO0_DIR=${OUTPUT_DIR}/"anno_0"
ANNO1_DIR=${OUTPUT_DIR}/"anno_1"
ANNO2_DIR=${OUTPUT_DIR}/"anno_2"
ANNO3_DIR=${OUTPUT_DIR}/"anno_3"
python3 evaluation/refytvos/eval_davis.py --results_path=${ANNO0_DIR}
python3 evaluation/refytvos/eval_davis.py --results_path=${ANNO1_DIR}
python3 evaluation/refytvos/eval_davis.py --results_path=${ANNO2_DIR}
python3 evaluation/refytvos/eval_davis.py --results_path=${ANNO3_DIR}
