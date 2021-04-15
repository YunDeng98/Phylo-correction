set -e

# Important hyperparameters
max_seqs=1024
max_sites=1024
armstrong_cutoff=8.0

# Irrelevant hyperparameters
n_process=32

# Input data directories
# Directory where the MSAs are found.
a3m_dir=a3m
# Directory where the pdb files are found
pdb_dir=pdb

# Output data directories
# Where the phylogenied will b e stored
tree_dir=trees_"$max_seqs"_seqs_"$max_sites"_sites
# Where the contacts will be stored
contact_dir=contacts_"$armstrong_cutoff"

# # First we need to generate the phylogenies
# pushd phylogeny_generation
# echo "Running phylogeny_generation.sh"
# bash phylogeny_generation.sh ../"$a3m_dir" "$max_seqs" "$max_sites" "$n_process" ../"$tree_dir"
# popd

# Generate the contacts
pushd contact_generation
echo "Running contact_generation.sh"
bash contact_generation.sh ../"$pdb_dir" "$n_process" ../"$contact_dir" "$armstrong_cutoff"
popd