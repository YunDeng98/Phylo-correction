set -e

a3m_dir=$1
maximum_parsimony_dir=$2
co_transitions_dir=$3
n_process=$4
expected_number_of_MSAs=$5
max_families=$6
contact_dir=$7

python3 co_transition_extraction.py --a3m_dir "$a3m_dir" --parsimony_dir "$maximum_parsimony_dir" --outdir "$co_transitions_dir" --n_process "$n_process" --expected_number_of_MSAs "$expected_number_of_MSAs" --max_families "$max_families" --contact_dir "$contact_dir"
