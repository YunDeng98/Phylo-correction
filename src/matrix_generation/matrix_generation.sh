set -e

a3m_dir=$1
transitions_dir=$2
matrices_dir=$3
n_process=$4
expected_number_of_MSAs=$5
max_families=$6
num_sites=$7

python3 matrix_generation.py --a3m_dir "$a3m_dir" --transitions_dir "$transitions_dir" --outdir "$matrices_dir" --n_process "$n_process" --expected_number_of_MSAs "$expected_number_of_MSAs" --max_families "$max_families" --num_sites "$num_sites"
