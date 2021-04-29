set -e

pdb_dir=$1
n_process=$2
contact_dir=$3
armstrong_cutoff=$4
expected_number_of_MSAs=$5
max_families=$6

echo "Generating Contact Matrices"
python3 contact_generation.py --armstrong_cutoff "$armstrong_cutoff" --pdb_dir "$pdb_dir" --outdir "$contact_dir" --n_process "$n_process" --expected_number_of_families "$expected_number_of_MSAs" --max_families "$max_families"

