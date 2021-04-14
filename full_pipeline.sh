set -e

# Directory where the MSAs are found.
a3m_dir=a3m
max_seqs=1024
max_sites=1024
n_process=16
tree_dir=trees_"$max_seqs"_seqs_"$max_sites"_sites

# First we need to generate the phylogenies
pushd phylogeny_generation
echo "Running phylogeny_generation.sh"
bash phylogeny_generation.sh ../"$a3m_dir" "$max_seqs" "$max_sites" "$n_process" ../"$tree_dir"
popd
