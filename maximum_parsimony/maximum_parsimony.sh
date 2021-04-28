set -e

g++ -std=c++17 -DACMTUYO -O3 -Wshadow -Wall  -Wextra -D_GLIBCXX_DEBUG -o maximum_parsimony maximum_parsimony.cpp

# ./maximum_parsimony test_data/tree.txt test_data/sequences.txt test_data/solution.txt
rm -rf ../maximum_parsimony_reconstructions
python3 maximum_parsimony.py --a3m_dir ../a3m --tree_dir ../trees --outdir ../maximum_parsimony_reconstructions --n_process 1 --expected_number_of_MSAs 1 --max_families 10