set -e

a3m_dir=$1
tree_dir=$2
a3m_simulated_dir=$3
contact_simulated_dir=$4
ancestral_states_simulated_dir=$5
n_process=$6
expected_number_of_MSAs=$7
max_families=$8
simulation_pct_interacting_positions=$9
Q1_ground_truth=${10}
Q2_ground_truth=${11}

python3 simulation.py \
--a3m_dir "$a3m_dir" \
--tree_dir "$tree_dir" \
--a3m_simulated_dir "$a3m_simulated_dir" \
--contact_simulated_dir "$contact_simulated_dir" \
--ancestral_states_simulated_dir "$ancestral_states_simulated_dir" \
--n_process "$n_process" \
--expected_number_of_MSAs "$expected_number_of_MSAs" \
--max_families "$max_families" \
--simulation_pct_interacting_positions "$simulation_pct_interacting_positions" \
--Q1_ground_truth "$Q1_ground_truth" \
--Q2_ground_truth "$Q2_ground_truth"
