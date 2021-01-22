#!/bin/bash

if [[ -z "$nthreads" ]]; then
    nthreads=$(($(nproc --all)/2))
fi
echo "Using $nthreads threads."

# Function to show attack results. Not used for scanning.
function show_rewards () {
    output_dir=$1
    echo "clean reward:"
    tail -n1 "${output_dir}/clean.log"
    echo "clean reward (deterministic):"
    tail -n1 "${output_dir}/clean_deterministic.log"
    echo "random attack reward:"
    tail -n1 "${output_dir}/random_attack.log"
    echo "random attack reward (deterministic):"
    tail -n1 "${output_dir}/random_attack_deterministic.log"
    echo "critic attack reward:"
    tail -n1 "${output_dir}/critic_attack.log"
    echo "critic attack reward (deterministic):"
    tail -n1 "${output_dir}/critic_attack_deterministic.log"
    echo "MAD attack reward:"
    tail -n1 "${output_dir}/mad_attack.log"
    echo "MAD attack reward (deterministic):"
    tail -n1 "${output_dir}/mad_attack_deterministic.log"
    echo "minimum RS attack reward:" 
    (for i in ${output_dir}/sarsa_*_attack.log; do tail -n1 $i | tr -d ',' | cut -d' ' -f 2; done) | sort -h | head -n 1
    echo "minimum RS attack reward (deterministic action):" 
    (for i in ${output_dir}/sarsa_*_attack_deterministic.log; do tail -n1 $i | tr -d ',' | cut -d' ' -f 2; done) | sort -h | head -n 1
    echo "minimum RS+MAD attack reward:" 
    (for i in ${output_dir}/sarsa-mad_*_attack.log; do tail -n1 $i | tr -d ',' | cut -d' ' -f 2; done) | sort -h | head -n 1
    echo "minimum RS+MAD attack reward (deterministic action):" 
    (for i in ${output_dir}/sarsa-mad_*_attack_deterministic.log; do tail -n1 $i | tr -d ',' | cut -d' ' -f 2; done) | sort -h | head -n 1
}

# Attack a single model.
function scan_attacks () {
    if [[ $# -lt 3 ]]; then
        echo "Usage: scan_attacks model_path config_path result_dir_path [semaphorename]"
        return 2
    fi
    model=$1
    config=$2
    output_dir=$3
    semaphorename=$4

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
    # semcmd="echo"  # for debugging

    # optionally, skip deterministic or stochastic attack evaluations.
    deter=false
    stochastic=false
    if [[ -z $ATTACK_MODEL_NO_DETERMINISTIC ]]; then
        deter=true
    fi
    if [[ -z $ATTACK_MODEL_NO_STOCHASTIC ]]; then
        stochastic=true
    fi

    $semcmd python test.py --config-path "$config" --load-model "$model" $FULL_ATTACK_EXTRAS ">" "${output_dir}/clean.log"
    $semcmd python test.py --config-path "$config" --load-model "$model" --deterministic $FULL_ATTACK_EXTRAS ">" "${output_dir}/clean_deterministic.log"
    for sarsa_eps in 0.02 0.05 0.1 0.15 0.2 0.3; do
        for sarsa_reg in 0.1 0.3 1.0 3.0 10.0; do
            echo ${sarsa_eps} ${sarsa_reg}
            # First train a sarsa model, then run stochastic and deterministic evaluation in parallel.
            $semcmd \
            python test.py --config-path "$config" --load-model "$model" --sarsa-model-path "${output_dir}/sarsa_${sarsa_eps}_${sarsa_reg}.model" --sarsa-enable --sarsa-eps ${sarsa_eps} --sarsa-reg ${sarsa_reg} --attack-method none $FULL_ATTACK_EXTRAS ">" "${output_dir}/sarsa_${sarsa_eps}_${sarsa_reg}_train.log" ";" \
            "$stochastic" "&&" $semcmd python test.py --config-path "$config" --load-model "$model" --attack-sarsa-network "${output_dir}/sarsa_${sarsa_eps}_${sarsa_reg}.model" --attack-method sarsa $FULL_ATTACK_EXTRAS  ">" "${output_dir}/sarsa_${sarsa_eps}_${sarsa_reg}_attack.log" "&" \
            "$deter" "&&" python test.py --config-path "$config" --load-model "$model" --attack-sarsa-network "${output_dir}/sarsa_${sarsa_eps}_${sarsa_reg}.model" --attack-method sarsa --deterministic $FULL_ATTACK_EXTRAS ">" "${output_dir}/sarsa_${sarsa_eps}_${sarsa_reg}_attack_deterministic.log"
        done
    done
    $stochastic && $semcmd python test.py --config-path "$config" --load-model "$model" --attack-method action $FULL_ATTACK_EXTRAS ">" "${output_dir}/mad_attack.log"
    $deter && $semcmd python test.py --config-path "$config" --load-model "$model" --attack-method action $FULL_ATTACK_EXTRAS --deterministic ">" "${output_dir}/mad_attack_deterministic.log"
    $stochastic && $semcmd python test.py --config-path "$config" --load-model "$model" --attack-method critic $FULL_ATTACK_EXTRAS ">" "${output_dir}/critic_attack.log"
    $deter && $semcmd python test.py --config-path "$config" --load-model "$model" --attack-method critic $FULL_ATTACK_EXTRAS --deterministic ">" "${output_dir}/critic_attack_deterministic.log"
    $stochastic && $semcmd python test.py --config-path "$config" --load-model "$model" --attack-method random $FULL_ATTACK_EXTRAS ">" "${output_dir}/random_attack.log"
    $deter && $semcmd python test.py --config-path "$config" --load-model "$model" --attack-method random $FULL_ATTACK_EXTRAS --deterministic ">" "${output_dir}/random_attack_deterministic.log"

    if [[ -z $ATTACK_MODEL_NO_WAIT ]]; then
        sem --wait $extra
        show_rewards $output_dir
    fi
}

# RS+MAD attack. Assuming we have trained and found the best sarsa model.
function scan_rs_mad_attacks () {
    if [[ $# -lt 3 ]]; then
        echo "Usage: scan_rs_mad_attacks model_path config_path result_dir_path [semaphorename]"
        return 2
    fi
    model=$1
    config=$2
    output_dir=$3
    semaphorename=$4

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
    # semcmd="echo"  # for debugging

    # optionally, skip deterministic or stochastic attack evaluations.
    deter=false
    stochastic=false
    if [[ -z $ATTACK_MODEL_NO_DETERMINISTIC ]]; then
        deter=true
    fi
    if [[ -z $ATTACK_MODEL_NO_STOCHASTIC ]]; then
        stochastic=true
    fi

    for mad_reg in 1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2 3e-2; do
        echo ${mad_reg}
        # Using a already trained sarsa model, sarsa_best.model. This link should be generated using mark_best.py
        "$stochastic" && $semcmd python test.py --config-path "$config" --load-model "$model" --attack-sarsa-network "${output_dir}/sarsa_best.model" --attack-method sarsa+action --attack-sarsa-action-ratio "${mad_reg}" $FULL_ATTACK_EXTRAS  ">" "${output_dir}/sarsa-mad_${mad_reg}_attack.log"
        "$deter" && $semcmd python test.py --config-path "$config" --load-model "$model" --attack-sarsa-network "${output_dir}/sarsa_best_deterministic.model" --attack-method sarsa+action --attack-sarsa-action-ratio "${mad_reg}" --deterministic $FULL_ATTACK_EXTRAS ">" "${output_dir}/sarsa-mad_${mad_reg}_attack_deterministic.log"
    done

    if [[ -z $ATTACK_MODEL_NO_WAIT ]]; then
        sem --wait $extra
        show_rewards $output_dir
    fi
}

# Attack a folder of models.
function scan_exp_folder () {
    config=$1
    folder=$2
    semaphorename=$3
    if [[ ! -f $config ]]; then
        echo "Config file $config not found!"
        return 1
    fi
    if [[ ! -d $folder ]]; then
        echo "experiment folder $folder not found!"
        return 1
    fi
    if [[ -z "$semaphorename" ]]; then
        semaphorename=$BASHPID
    fi
    echo "Attack folder $folder with config $config using semaphore $semaphorename"
    list=$(find $folder -maxdepth 1 -mindepth 1 -type d)
    count=$(find $folder -maxdepth 1 -mindepth 1 -type d | wc -l)
    # Set this flag so that the sem in attack scan does not wait.
    export ATTACK_MODEL_NO_WAIT=1
    i=0
    for f in $list; do
        uuid=$(basename $f)
        ((i=i+1))
        # check if the folder name is a uuid.
        if [[ $uuid =~ ^\{?[A-F0-9a-f]{8}-[A-F0-9a-f]{4}-[A-F0-9a-f]{4}-[A-F0-9a-f]{4}-[A-F0-9a-f]{12}\}?$ ]]; then
            if [[ ! -f "$f/NO_ATTACK" ]]; then
                echo "Processing $f, progress $i / $count"
                output_dir=$f/attack_scan
                model=${output_dir}/model
                mkdir -p $output_dir
                # First, extract model.
                python get_best_pickle.py --output $model $f
                # Then launch parallel attack.
                if [[ -z $RUN_MAD_ATTACK ]]; then
                    scan_attacks $model $config $output_dir $semaphorename
                else
                    scan_rs_mad_attacks $model $config $output_dir $semaphorename
                fi
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

# Example:
# scan_attacks models/model-sappo-convex-humanoid.model config_humanoid_robust_ppo_convex.json sarsa_humanoid_sappo-convex
# scan_exp_folder config_ant_vanilla_ppo.json adv_ppo_ant/agents

