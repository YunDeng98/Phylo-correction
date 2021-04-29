set -e

a3m_dir=$1
tree_dir=$2
maximum_parsimony_dir=$3
n_process=$4
expected_number_of_MSAs=$5
max_families=$6

g++ -std=c++17 -O3 -Wshadow -Wall  -Wextra -D_GLIBCXX_DEBUG -o maximum_parsimony maximum_parsimony.cpp
# ./maximum_parsimony test_data/tree.txt test_data/sequences.txt test_data/solution.txt

python3 maximum_parsimony.py --a3m_dir "$a3m_dir" --tree_dir "$tree_dir" --outdir "$maximum_parsimony_dir" --n_process "$n_process" --expected_number_of_MSAs "$expected_number_of_MSAs" --max_families "$max_families"
