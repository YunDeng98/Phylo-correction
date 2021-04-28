set -e

# Important hyperparameters
max_seqs=1024
max_sites=1024
armstrong_cutoff=8.0

# Irrelevant hyperparameters
n_process=16

# Input data directories
# Directory where the MSAs are found.
a3m_dir=a3m
# Directory where the pdb files are found
pdb_dir=pdb

# Output data directories
# Where the phylogenies will be stored
tree_dir=trees_"$max_seqs"_seqs_"$max_sites"_sites
# Where the contacts will be stored
contact_dir=contacts_"$armstrong_cutoff"
# Where the maximum parsimony reconstructions will be stored
maximum_parsimony_dir=maximum_parsimony_"$max_seqs"_seqs_"$max_sites"_sites

# First we need to generate the phylogenies
pushd phylogeny_generation
echo "Running phylogeny_generation.sh"
bash phylogeny_generation.sh ../"$a3m_dir" "$max_seqs" "$max_sites" "$n_process" ../"$tree_dir"
popd

# Generate the contacts
pushd contact_generation
echo "Running contact_generation.sh"
bash contact_generation.sh ../"$pdb_dir" "$n_process" ../"$contact_dir" "$armstrong_cutoff"
popd

# Generate the maximum parsimony reconstructions
pushd maximum_parsimony
echo "Running maximum_parsimony.sh"
bash maximum_parsimony.sh ../"$a3m_dir" ../"$tree_dir" ../"$maximum_parsimony_dir" "$n_process"
popd
