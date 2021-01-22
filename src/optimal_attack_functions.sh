#!/bin/bash

if [[ -z "$nthreads" ]]; then
    nthreads=$(($(nproc --all)/2))
fi
echo "Using $nthreads threads."

# Attack a single model.
function scan_attacks () {
    trap "kill 0" SIGINT  # exit cleanly when pressing control+C

    if [[ $# -lt 3 ]]; then
        echo "Usage: scan_attacks model_path config_path result_dir_path [semaphorename]"
        return 2
    fi
    model=$1
    config=$2
    output_dir=$3
    agent_model=$4
    semaphorename=$5

    if [[ -z "$semaphorename" ]]; then
        extra="--semaphorename $BASHPID"
    else
        extra="--semaphorename $semaphorename"
    fi
    sqlitefile=${output_dir}/results.sqlite

    mkdir -p ${output_dir}
    echo "Extra attack parameters: ${ATTACK_EXTRAS}"
    FULL_ATTACK_EXTRAS="${ATTACK_EXTRAS} --sqlite-path ${sqlitefile} --early-terminate"
    echo "Scanning results will be saved to ${output_dir}"
    semcmd="sem -j $nthreads $extra"

    $semcmd python test.py --config-path "$config" --attack-advpolicy-network "$model" --load-model "$agent_model" --attack-method advpolicy --deterministic $FULL_ATTACK_EXTRAS ">" "${output_dir}/optatk_deterministic.log" 

    if [[ -z $ATTACK_MODEL_NO_WAIT ]]; then
        sem --wait $extra
    fi
}

# Attack a folder of models.
function scan_exp_folder () {
    config=$1
    folder=$2
    agent_model=$3
    semaphorename=$4
    if [[ ! -f $config ]]; then
        echo "Config file $config not found!"
        return 1
    fi
    if [[ ! -d $folder ]]; then
        echo "experiment folder $folder not found!"
        return 1
    fi
    if [[ ! -f $agent_model ]]; then
        echo "agent model  $agent_model not found!"
        return 1
    fi
    if [[ -z "$semaphorename" ]]; then
        semaphorename=$BASHPID
    fi
    echo "Attack folder $folder with config $config with agent model $agent_model using semaphore $semaphorename"
    list=$(find $folder -maxdepth 1 -mindepth 1 -type d)
    # Set this flag so that the sem in attack scan does not wait.
    export ATTACK_MODEL_NO_WAIT=1
    for f in $list; do
        uuid=$(basename $f)
        # check if the folder name is a uuid.
        if [[ $uuid =~ ^\{?[A-F0-9a-f]{8}-[A-F0-9a-f]{4}-[A-F0-9a-f]{4}-[A-F0-9a-f]{4}-[A-F0-9a-f]{12}\}?$ ]]; then
            if [[ ! -f "$f/NO_ATTACK" ]]; then
                echo "Processing $f"
                output_dir=$f/attack_scan
                model=${output_dir}/model
                mkdir -p $output_dir
                # First, extract model.
                python get_best_pickle.py --output $model $f
                # Then launch parallel attack.
                scan_attacks $model $config $output_dir $agent_model $semaphorename
            else
                echo "Skipping $f because NO_ATTACK flag is set"
            fi
        else
            echo "Skipping $f"
        fi
    done
    if [[ -z $ATTACK_FOLDER_NO_WAIT ]]; then
        sem --wait --semaphorename $semaphorename
    fi
}

