set -e

a3m_dir=$1
max_seqs=$2
max_sites=$3
n_process=$4
tree_dir=$5
expected_number_of_MSAs=$6
max_families=$7
rate_matrix=$8

echo "Running install_fast_tree.sh"
bash install_fast_tree.sh

# echo "Running fast_tree_benchmark.py"
# python3 fast_tree_benchmark.py --a3m_dir "$a3m_dir" --max_seqs "$max_seqs" --max_sites "$max_sites" --protein_family_name 1twf_1_B --outdir fast_tree_benchmark_output

# echo "Running fast_tree_benchmark_extrapolation.py"
# python3 fast_tree_benchmark_extrapolation.py --a3m_dir "$a3m_dir" --max_seqs "$max_seqs" --max_sites "$max_sites" --outdir fast_tree_benchmark_extrapolation_output --grid_path ./fast_tree_benchmark_output/benchmark_results.csv

echo "Generating FastTree Phylogenies"
python3 generate_fast_tree_phylogenies.py --a3m_dir "$a3m_dir" --outdir "$tree_dir" --n_process "$n_process" --expected_number_of_MSAs "$expected_number_of_MSAs" --max_seqs "$max_seqs" --max_sites "$max_sites" --max_families "$max_families" --rate_matrix "$rate_matrix"
