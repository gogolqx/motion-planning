#!/bin/bash

prefix="scenarioclustering_"
suffix="_run_1"
all_running=""

running () {
    first="$(docker inspect -f '{{.State.Running}}' ${prefix}$1${suffix} 2>/dev/null)"
    second="$(docker inspect -f '{{.State.Running}}' ${prefix}$2${suffix} 2>/dev/null)"
    third="$(docker inspect -f '{{.State.Running}}' ${prefix}$3${suffix} 2>/dev/null)"
    fourth="$(docker inspect -f '{{.State.Running}}' ${prefix}$4${suffix} 2>/dev/null)"

    if [ "${first}"  = "true" ] || [ "${second}" = "true" ] || [ "${third}" = "true" ] || [ "${fourth}" = "true" ]; then
        all_running="true"
    else
        all_running="false"
    fi
}

docker image rm clustering_base
docker build --tag clustering_base .

mkdir -p results/dhw-small-all
mkdir -p results/dhw-small-dhw
mkdir -p results/dhw-small-thw
mkdir -p results/dhw-small-ttc
mkdir -p results/thw-small-all
mkdir -p results/thw-small-dhw
mkdir -p results/thw-small-thw
mkdir -p results/thw-small-ttc
mkdir -p results/ttc-small-all
mkdir -p results/ttc-small-dhw
mkdir -p results/ttc-small-thw
mkdir -p results/ttc-small-ttc
mkdir -p results/dhw-big-all
mkdir -p results/dhw-big-dhw
mkdir -p results/dhw-big-thw
mkdir -p results/dhw-big-ttc
mkdir -p results/thw-big-all
mkdir -p results/thw-big-dhw
mkdir -p results/thw-big-thw
mkdir -p results/thw-big-ttc
mkdir -p results/ttc-big-all
mkdir -p results/ttc-big-dhw
mkdir -p results/ttc-big-thw
mkdir -p results/ttc-big-ttc

# Run all clusters back to back 4 at a time.
tput setaf 2; echo "Running clusterings for Min DHW Small Dataset"
tput sgr0
docker-compose run -d --user $(id -u):$(id -g) dhw-small-all
docker-compose run -d --user $(id -u):$(id -g) dhw-small-dhw
docker-compose run -d --user $(id -u):$(id -g) dhw-small-thw
docker-compose run -d --user $(id -u):$(id -g) dhw-small-ttc
running "dhw-small-all" "dhw-small-dhw" "dhw-small-thw" "dhw-small-ttc"
echo ""

while [[ "${all_running}" = "true" ]]
do
    sleep 10
    echo "Waiting for the batch to finish..."
    running "dhw-small-all" "dhw-small-dhw" "dhw-small-thw" "dhw-small-ttc"
done

docker container rm "${prefix}dhw-small-all${suffix}"
docker container rm "${prefix}dhw-small-dhw${suffix}"
docker container rm "${prefix}dhw-small-thw${suffix}"
docker container rm "${prefix}dhw-small-ttc${suffix}"



tput setaf 2; echo "Running clusterings for Min THW Small Dataset"
tput sgr0
docker-compose run -d --user $(id -u):$(id -g) thw-small-all
docker-compose run -d --user $(id -u):$(id -g) thw-small-dhw
docker-compose run -d --user $(id -u):$(id -g) thw-small-thw
docker-compose run -d --user $(id -u):$(id -g) thw-small-ttc
running "thw-small-all" "thw-small-dhw" "thw-small-thw" "thw-small-ttc"
echo ""

while [[ "${all_running}" = "true" ]]
do
    sleep 10
    echo "Waiting for the batch to finish..."
    running "thw-small-all" "thw-small-dhw" "thw-small-thw" "thw-small-ttc"
done

docker container rm "${prefix}thw-small-all${suffix}"
docker container rm "${prefix}thw-small-dhw${suffix}"
docker container rm "${prefix}thw-small-thw${suffix}"
docker container rm "${prefix}thw-small-ttc${suffix}"



tput setaf 2; echo "Running clusterings for Min TTC Small Dataset"
tput sgr0
docker-compose run -d --user $(id -u):$(id -g) ttc-small-all
docker-compose run -d --user $(id -u):$(id -g) ttc-small-dhw
docker-compose run -d --user $(id -u):$(id -g) ttc-small-thw
docker-compose run -d --user $(id -u):$(id -g) ttc-small-ttc
running "ttc-small-all" "ttc-small-dhw" "ttc-small-thw" "ttc-small-ttc"
echo ""

while [[ "${all_running}" = "true" ]]
do
    sleep 10
    echo "Waiting for the batch to finish..."
    running "ttc-small-all" "ttc-small-dhw" "ttc-small-thw" "ttc-small-ttc"
done

docker container rm "${prefix}ttc-small-all${suffix}"
docker container rm "${prefix}ttc-small-dhw${suffix}"
docker container rm "${prefix}ttc-small-thw${suffix}"
docker container rm "${prefix}ttc-small-ttc${suffix}"



#tput setaf 2; echo "Running clusterings for Min DHW Big Dataset"
#tput sgr0
#docker-compose run -d --user $(id -u):$(id -g) dhw-big-all
#docker-compose run -d --user $(id -u):$(id -g) dhw-big-dhw
#docker-compose run -d --user $(id -u):$(id -g) dhw-big-thw
#docker-compose run -d --user $(id -u):$(id -g) dhw-big-ttc
#running "dhw-big-all" "dhw-big-dhw" "dhw-big-thw" "dhw-big-ttc"
#echo ""
#
#while [[ "${all_running}" = "true" ]]
#do
#    sleep 10
#    echo "Waiting for the batch to finish..."
#    running "dhw-big-all" "dhw-big-dhw" "dhw-big-thw" "dhw-big-ttc"
#done
#
#docker container rm "${prefix}dhw-big-all${suffix}"
#docker container rm "${prefix}dhw-big-dhw${suffix}"
#docker container rm "${prefix}dhw-big-thw${suffix}"
#docker container rm "${prefix}dhw-big-ttc${suffix}"
#
#
#
#tput setaf 2; echo "Running clusterings for Min THW Big Dataset"
#tput sgr0
#docker-compose run -d --user $(id -u):$(id -g) thw-big-all
#docker-compose run -d --user $(id -u):$(id -g) thw-big-dhw
#docker-compose run -d --user $(id -u):$(id -g) thw-big-thw
#docker-compose run -d --user $(id -u):$(id -g) thw-big-ttc
#running "thw-big-all" "thw-big-dhw" "thw-big-thw" "thw-big-ttc"
#echo ""
#
#while [[ "${all_running}" = "true" ]]
#do
#    sleep 10
#    echo "Waiting for the batch to finish..."
#    running "thw-big-all" "thw-big-dhw" "thw-big-thw" "thw-big-ttc"
#done
#
#docker container rm "${prefix}thw-big-all${suffix}"
#docker container rm "${prefix}thw-big-dhw${suffix}"
#docker container rm "${prefix}thw-big-thw${suffix}"
#docker container rm "${prefix}thw-big-ttc${suffix}"
#
#
#
#tput setaf 2; echo "Running clusterings for Min TTC Big Dataset"
#tput sgr0
#docker-compose run -d --user $(id -u):$(id -g) ttc-big-all
#docker-compose run -d --user $(id -u):$(id -g) ttc-big-dhw
#docker-compose run -d --user $(id -u):$(id -g) ttc-big-thw
#docker-compose run -d --user $(id -u):$(id -g) ttc-big-ttc
#running "ttc-big-all" "ttc-big-dhw" "ttc-big-thw" "ttc-big-ttc"
#echo ""
#
#while [[ "${all_running}" = "true" ]]
#do
#    sleep 10
#    echo "Waiting for the batch to finish..."
#    running "ttc-big-all" "ttc-big-dhw" "ttc-big-thw" "ttc-big-ttc"
#done
#
#docker container rm "${prefix}ttc-big-all${suffix}"
#docker container rm "${prefix}ttc-big-dhw${suffix}"
#docker container rm "${prefix}ttc-big-thw${suffix}"
#docker container rm "${prefix}ttc-big-ttc${suffix}"

tput setaf 2; echo "Finished all clustering batches!"
tput sgr0
