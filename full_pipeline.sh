set -e

# Important hyperparameters
max_seqs=1024
max_sites=1024
armstrong_cutoff=8.0

# Irrelevant hyperparameters
n_process=6
expected_number_of_MSAs=6
max_families=4

# Input data directories
# Directory where the MSAs are found.
a3m_dir=a3m_test_large
# # Directory where the pdb files are found
# pdb_dir=pdb

# Output data directories
# Where the phylogenies will be stored
tree_dir=trees_"$max_seqs"_seqs_"$max_sites"_sites
# Where the contacts will be stored
contact_dir=contacts_"$armstrong_cutoff"
# Where the maximum parsimony reconstructions will be stored
maximum_parsimony_dir=maximum_parsimony_"$max_seqs"_seqs_"$max_sites"_sites
# Where the transitions obtained from the maximum parsimony phylogenies will be stored
transitions_dir=transitions_"$max_seqs"_seqs_"$max_sites"_sites
# Where the transition matrices obtained by quantizing transition edges will be stored
matrices_dir=matrices_"$max_seqs"_seqs_"$max_sites"_sites

# First we need to generate the phylogenies
pushd phylogeny_generation
echo "Running phylogeny_generation.sh"
bash phylogeny_generation.sh ../"$a3m_dir" "$max_seqs" "$max_sites" "$n_process" ../"$tree_dir" "$expected_number_of_MSAs" "$max_families"
popd

# # Generate the contacts
# pushd contact_generation
# echo "Running contact_generation.sh"
# bash contact_generation.sh ../"$pdb_dir" "$n_process" ../"$contact_dir" "$armstrong_cutoff" "$expected_number_of_MSAs" "$max_families"
# popd

# Generate the maximum parsimony reconstructions
pushd maximum_parsimony
echo "Running maximum_parsimony.sh"
bash maximum_parsimony.sh ../"$a3m_dir" ../"$tree_dir" ../"$maximum_parsimony_dir" "$n_process" "$expected_number_of_MSAs" "$max_families"
popd

# Generate transitions
pushd transition_extraction
echo "Running transition_extraction.sh"
bash transition_extraction.sh ../"$a3m_dir" ../"$maximum_parsimony_dir" ../"$transitions_dir" "$n_process" "$expected_number_of_MSAs" "$max_families"
popd

# Generate transition matrices
pushd matrix_generation
echo "Running matrix_generation.sh"
bash matrix_generation.sh ../"$a3m_dir" ../"$transitions_dir" ../"$matrices_dir" "$n_process" "$expected_number_of_MSAs" "$max_families"
popd
