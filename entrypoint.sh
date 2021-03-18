#!/bin/bash

rf_args="-d ./features.csv"

if [[ ! -z "${FEATURES}" ]]; then
    rf_args="${rf_args} -f ${FEATURES}"
fi

if [[ ! -z "${NUMTREES}" ]]; then
    rf_args="${rf_args} -nt ${NUMTREES}"
fi

if [[ ! -z "${TERMLEAVES}" ]]; then
    rf_args="${rf_args} -tl ${TERMLEAVES}"
fi

if [[ ! -z "${MINGINI}" ]]; then
    rf_args="${rf_args} -ming ${MINGINI}"
fi

if [[ ! -z "${MAXGINI}" ]]; then
    rf_args="${rf_args} -maxg ${MAXGINI}"
fi

if [[ ! -z "${NUMCLUS}" ]]; then
    rf_args="${rf_args} -nc ${NUMCLUS}"
fi

if [[ ! -z "${LENDATA}" ]]; then
    rf_args="${rf_args} -l ${LENDATA}"
fi

if [[ ! -z "${REDUCE}" ]]; then
    rf_args="${rf_args} -r ${REDUCE}"
fi

if [[ ! -z "${DEBUGRF}" ]]; then
    rf_args="${rf_args} --debug"
fi

tput setaf 2; echo "Random Forest parameters:"
tput sgr0
echo $rf_args
echo ""

Fecho "Running Random Forest..."
tput sgr0
python3.7 run.py $rf_args
tput setaf 2; echo "Done!"
tput sgr0
echo ""



declare -a CLUSTERING_ARRAY="${CLUSTERTYPES}"
num_clustering=${#CLUSTERING_ARRAY[@]}
tput setaf 2; echo "Running $num_clustering different clustering on proximity matrix..."
tput sgr0
echo ""

for ((i=0; i<${num_clustering}; i++)); do
    tput setaf 2; echo "Running the clustering number: $i ..."
    tput sgr0
    echo ""

    c_args=" ${CLUSTERING_ARRAY[$i]} ./features.csv"

    matrix_pkl_file=$(ls ./outputs/matrix_*.pkl | sort -V | tail -n 1)
    c_args="${c_args} ${matrix_pkl_file}"

    if [[ ! -z "${EPSILONS}" ]]; then
        declare -a EPSILON_ARRAY="${EPSILONS}"
        if [[ ! "${EPSILON_ARRAY[$i]}" = "EMPTY" ]]; then
            c_args="${c_args} -e ${EPSILON_ARRAY[$i]}"
        fi
    fi

    if [[ ! -z "${NEIGHBOURS}" ]]; then
        declare -a NEIGHBOR_ARRAY="${NEIGHBOURS}"
        if [[ ! "${NEIGHBOR_ARRAY[$i]}" = "EMPTY" ]]; then
            c_args="${c_args} -n ${NEIGHBOR_ARRAY[$i]}"
        fi
    fi

    if [[ ! -z "${CNUMCLUSTERS}" ]]; then
        declare -a CNUMCLUSTER_ARRAY="${CNUMCLUSTERS}"
        if [[ ! "${CNUMCLUSTER_ARRAY[$i]}" = "EMPTY" ]]; then
            c_args="${c_args} -n ${CNUMCLUSTER_ARRAY[$i]}"
        fi
    fi

    if [[ ! -z "${MAXDISTS}" ]]; then
        declare -a MAXDIST_ARRAY="${MAXDISTS}"
        if [[ ! "${MAXDIST_ARRAY[$i]}" = "EMPTY" ]]; then
            c_args="${c_args} -d ${MAXDIST_ARRAY[$i]}"
        fi
    fi

    if [[ ! -z "${METRICS}" ]]; then
        declare -a METRIC_ARRAY="${METRICS}"
        if [[ ! "${METRIC_ARRAY[$i]}" = "EMPTY" ]]; then
            c_args="${c_args} -m ${METRIC_ARRAY[$i]}"
        fi
    fi

    tput setaf 2; echo "Clustering parameters:"
    tput sgr0
    echo $c_args
    echo ""

    tput setaf 2; echo "Running Clustering..."
    tput sgr0
    python3.7 cluster.py $c_args
    tput setaf 2; echo "Done!"
    tput sgr0
    echo ""
done

tput setaf 2; echo "Finished!"
tput sgr0