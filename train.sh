#!/bin/bash

mode=${1}
algo=${2:-"dqn"}
den=${3:-"10x10"}

total_timesteps=300000
log_dir="logs/${algo}_latest"
device="cuda"

mkdir -p ${log_dir}

if [ "${mode}" == "data_gen" ]; then
    echo "Starting data generation mode..."
    python -m utils.dataset --algo ${algo} --timesteps ${total_timesteps} --output_dir ${log_dir}
elif [ "${mode}" == "train" ]; then
    echo "Starting training mode..."
    if [ "${den}" == "10x10" ]; then
        train_glob="dataset5k/10x10_*.jsp"
        val_glob="benchmarks/validation/10x10_*.jsp"
    else
        train_glob=""
        val_glob=""
    fi
    python -m agents.train_sb3 --algo ${algo} --total_timesteps ${total_timesteps} --logdir ${log_dir} --train_glob "${train_glob}" --val_glob "${val_glob}" --n_envs 8 --net_size 128 --device ${device}
    if [ -n "${val_glob}" ]; then
        val_instance=$(ls ${val_glob} | head -n 1)
        echo "Evaluating on ${val_instance} and plotting Gantt chart..."
        model_file="${log_dir}/${algo}_final.zip"
        if [ ! -f "${model_file}" ]; then
            model_file=$(ls -t ${log_dir}/*.zip 2>/dev/null | head -n 1)
        fi
        if [ -n "${model_file}" ]; then
            echo "Using model: ${model_file}"
            python -m utils.eval --algo ${algo} --model_path ${model_file} --instance ${val_instance}
        else
            echo "No model checkpoint found in ${log_dir}. Skipping evaluation."
        fi
    fi
elif [ "${mode}" == "eval" ]; then
    echo "Starting evaluation mode..."
    val_glob="benchmarks/validation/10x10_1.jsp"
    model_file="${log_dir}/${algo}_final.zip"
    if [ ! -f "${model_file}" ]; then
        model_file=$(ls -t ${log_dir}/*.zip 2>/dev/null | head -n 1)
    fi
    if [ -n "${model_file}" ]; then
        echo "Using model: ${model_file}"
        for instance in ${val_glob}; do
            echo "Evaluating on ${instance} and plotting Gantt chart..."
            python -m utils.eval --algo ${algo} --model_path ${model_file} --instance ${instance}
        done
    else
        echo "No model checkpoint found in ${log_dir}. Skipping evaluation."
    fi
fi