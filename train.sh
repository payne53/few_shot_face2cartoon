#!/bin/bash

CUR_DIR=$(dirname $0)
# shellcheck disable=SC2164
cd ${CUR_DIR}

DATA_PATH='hdfs://haruna/home/byte_arnold_hl_vc/zhuangnan/photo2cartoon/dataset.tar'
echo "==> Copy '${DATA_PATH}' to '${CUR_DIR}'"
hadoop fs -copyToLocal "${DATA_PATH}" "${CUR_DIR}"

PRETRAIN_PATH='hdfs://haruna/home/byte_arnold_hl_vc/zhuangnan/photo2cartoon/pretrained_weights'
echo "==> Copy '${PRETRAIN_PATH}' to '${CUR_DIR}'"
hadoop fs -copyToLocal "${PRETRAIN_PATH}" "${CUR_DIR}"

echo "extract dataset"
tar -xf dataset.tar dataset

# shellcheck disable=SC2068
python3 train.py ${@}

hdfs dfs -put -f ./experiment/* hdfs://haruna/home/byte_arnold_hl_vc/zhuangnan/photo2cartoon/experiment/

echo "finished"