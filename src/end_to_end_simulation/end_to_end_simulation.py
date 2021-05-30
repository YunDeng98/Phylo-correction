import os

from src.pipeline import Pipeline
from src.simulation import Simulator


class EndToEndSimulator:
    r"""
    Given a Pipeline, the EndToEndSimulator uses ground-truth rate matrices to
    simulate contact maps and MSAs using the trees of that pipeline, then
    re-runs the pipeline on the simulated data. This allows one to test the
    quality of the Pipeline as the estimator, and evaluate design decisions
    such as bias of maximum parsimony. The pipeline is specified in __init__()
    and only run when the run() method is called.

    Args:
        outdir: Directory where the simulated data, as well as the output of
            running the pipeline on the simulated data, will be found.
        pipeline: Pipeline to perform end-to-end simulation on.
        simulation_pct_interacting_positions: What percent of sites will be
            considered contacting.
        Q1_ground_truth: Ground-truth single-site rate matrix.
        Q2_ground_truth: Ground-truth co-evolution rate matrix.
        fast_tree_rate_matrix: When the pipeline is run on the simulated data,
            this rate matrix will be used in FastTree instead. This is helpful
            because one might want to test the pipeline on data that was
            generated with a single-site model (Q1_ground_truth) that is
            different from standard amino-acid matrices. In that case,
            the phylogeny reconstruction step should use a matrix that
            aligns with Q1_ground_truth instead.
    """

    def __init__(
        self,
        outdir: str,
        pipeline: Pipeline,
        simulation_pct_interacting_positions: float,
        Q1_ground_truth: str,
        Q2_ground_truth: str,
        fast_tree_rate_matrix: str,
    ):
        self.outdir = outdir
        self.pipeline = pipeline
        self.simulation_pct_interacting_positions = simulation_pct_interacting_positions
        self.Q1_ground_truth = Q1_ground_truth
        self.Q2_ground_truth = Q2_ground_truth
        self.fast_tree_rate_matrix = fast_tree_rate_matrix

    def run(self):
        outdir = self.outdir
        pipeline = self.pipeline
        simulation_pct_interacting_positions = self.simulation_pct_interacting_positions
        Q1_ground_truth = self.Q1_ground_truth
        Q2_ground_truth = self.Q2_ground_truth
        fast_tree_rate_matrix = self.fast_tree_rate_matrix
        a3m_simulated_dir = os.path.join(outdir, "a3m_simulated")
        contact_simulated_dir = os.path.join(outdir, "contacts_simulated")
        ancestral_states_simulated_dir = os.path.join(outdir, "ancestral_states_simulated")

        simulator = Simulator(
            a3m_dir=pipeline.a3m_dir,
            tree_dir=pipeline.tree_dir,
            a3m_simulated_dir=a3m_simulated_dir,
            contact_simulated_dir=contact_simulated_dir,
            ancestral_states_simulated_dir=ancestral_states_simulated_dir,
            n_process=pipeline.n_process,
            expected_number_of_MSAs=pipeline.expected_number_of_MSAs,
            max_families=pipeline.max_families,
            simulation_pct_interacting_positions=simulation_pct_interacting_positions,
            Q1_ground_truth=Q1_ground_truth,
            Q2_ground_truth=Q2_ground_truth,
        )
        simulator.run()

        # Run pipeline on simulated data.
        end_to_end_pipeline_on_simulated_data = Pipeline(
            outdir=os.path.join(outdir, pipeline.outdir),
            max_seqs=pipeline.max_seqs,
            max_sites=pipeline.max_sites,
            armstrong_cutoff=None,
            rate_matrix=fast_tree_rate_matrix,
            n_process=pipeline.n_process,
            expected_number_of_MSAs=pipeline.max_families,  # BC we only generated max_families MSAs!
            max_families=pipeline.max_families,
            a3m_dir=a3m_simulated_dir,
            pdb_dir=None,
            precomputed_contact_dir=contact_simulated_dir,
        )
        end_to_end_pipeline_on_simulated_data.run()
