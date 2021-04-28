set -e

g++ -std=c++17 -DACMTUYO -O3 -Wshadow -Wall  -Wextra -D_GLIBCXX_DEBUG -o maximum_parsimony maximum_parsimony.cpp

./maximum_parsimony test_data/tree.txt test_data/sequences.txt test_data/solution.txt
