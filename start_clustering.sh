# Script for starting a single clustering
# Example usage:
#   bash start_clustering.sh dhw-small-all

prefix="scenarioclustering_"
suffix="_run_1"
service_name=$1

docker image rm clustering_base
docker build --tag clustering_base .

# Run all clusters back to back 4 at a time.
tput setaf 2; echo "Running clusterings for service: ${1}"
tput sgr0
mkdir -p results/$1
docker-compose run --user $(id -u):$(id -g) "${1}"
echo ""

docker container rm "${prefix}${1}${suffix}"
