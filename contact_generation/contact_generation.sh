set -e

pdb_dir=$1
n_process=$2
contact_dir=$3
armstrong_cutoff=$4

echo "Generating Contact Matrices"
python3 contact_generation.py --armstrong_cutoff "$armstrong_cutoff" --pdb_dir "$pdb_dir" --outdir "$contact_dir" --n_process "$n_process" --expected_number_of_families 15051 --max_families 100000000

