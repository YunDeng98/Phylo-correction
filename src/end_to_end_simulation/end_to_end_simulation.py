import logging
import os
import time

from src.pipeline import Pipeline
from src.simulation import Simulator

from typing import Optional


class EndToEndSimulator:
    r"""
    Perform End-To-End simulation of a rate matrix estimation pipeline.

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
        simulate_end_to_end: If to run end-to-end simulation, which starts from
            only the MSAs and contact maps.
        simulate_from_trees_wo_ancestral_states: If to run simulation starting
            from the MSAs and contact maps and *ground truth trees wo/ancestral states*.
        simulate_from_trees_w_ancestral_states: If to run simulation starting
            from the MSAs and contact maps and *ground truth trees w/ancestral states*.

    Attributes:
        time_***: The time taken for each step of the end-to-end simulation.
    """

    def __init__(
        self,
        outdir: str,
        pipeline: Pipeline,
        simulation_pct_interacting_positions: float,
        Q1_ground_truth: str,
        Q2_ground_truth: str,
        fast_tree_rate_matrix: str,
        simulate_end_to_end: Optional[bool],
        simulate_from_trees_wo_ancestral_states: Optional[bool],
        simulate_from_trees_w_ancestral_states: Optional[bool],
        use_cached: bool,
    ):
        self.outdir = outdir
        self.pipeline = pipeline
        self.simulation_pct_interacting_positions = simulation_pct_interacting_positions
        self.Q1_ground_truth = Q1_ground_truth
        self.Q2_ground_truth = Q2_ground_truth
        self.fast_tree_rate_matrix = fast_tree_rate_matrix
        self.simulate_end_to_end = simulate_end_to_end
        self.simulate_from_trees_wo_ancestral_states = simulate_from_trees_wo_ancestral_states
        self.simulate_from_trees_w_ancestral_states = simulate_from_trees_w_ancestral_states
        self.use_cached = use_cached

    def run(self):
        logger = logging.getLogger("phylo_correction.end_to_end_simulation")
        outdir = self.outdir
        pipeline = self.pipeline
        simulation_pct_interacting_positions = self.simulation_pct_interacting_positions
        Q1_ground_truth = self.Q1_ground_truth
        Q1_ground_truth_name = str(Q1_ground_truth).split('/')[-1]
        Q2_ground_truth = self.Q2_ground_truth
        Q2_ground_truth_name = str(Q2_ground_truth).split('/')[-1]
        fast_tree_rate_matrix = self.fast_tree_rate_matrix
        a3m_simulated_dir = os.path.join(outdir, f"a3m_simulated_{pipeline.max_seqs}_seqs_{pipeline.max_sites}_sites_{pipeline.rate_matrix_name}_RM__{Q1_ground_truth_name}_Q1_{Q2_ground_truth_name}_Q2_{simulation_pct_interacting_positions}_pct")
        contact_simulated_dir = os.path.join(outdir, f"contacts_simulated_{pipeline.max_seqs}_seqs_{pipeline.max_sites}_sites_{pipeline.rate_matrix_name}_RM_{simulation_pct_interacting_positions}_pct")
        ancestral_states_simulated_dir = os.path.join(outdir, f"ancestral_states_simulated_{pipeline.max_seqs}_seqs_{pipeline.max_sites}_sites_{pipeline.rate_matrix_name}_RM__{Q1_ground_truth_name}_Q1_{Q2_ground_truth_name}_Q2_{simulation_pct_interacting_positions}_pct")
        simulate_end_to_end = self.simulate_end_to_end
        simulate_from_trees_wo_ancestral_states = self.simulate_from_trees_wo_ancestral_states
        simulate_from_trees_w_ancestral_states = self.simulate_from_trees_w_ancestral_states
        use_cached = self.use_cached

        if os.path.exists(outdir):
            raise ValueError(f"Output directory {outdir} already exists. Please choose a different one!")

        if not os.path.exists(pipeline.tree_dir):
            raise ValueError("pipeline's trees do not exist! Have you already run the pipeline?")
        if not os.path.exists(pipeline.maximum_parsimony_dir):
            raise ValueError("pipeline's maximum_parsimony_dir does not exist! Have you already run the pipeline?")
        if (
            pipeline.precomputed_contact_dir is not None
            or pipeline.precomputed_tree_dir is not None
            or pipeline.precomputed_maximum_parsimony_dir is not None
        ):
            raise ValueError(
                "Trying to perform simulation on a pipeline that already uses simulated data! This is"
                " certainly a user bug."
            )

        os.makedirs(outdir)  # We know this doesn't exist yet bc we are pedantic.
        pipeline_info_path = os.path.join(outdir, 'pipeline_info.txt')
        with open(pipeline_info_path, "w") as pipeline_info_file:
            pipeline_info_file.write(str(pipeline))

        t_start = time.time()
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
            use_cached=use_cached,
        )
        simulator.run()
        self.time_Simulator = time.time() - t_start
        logger.info(f"time_Simulator = {self.time_Simulator}")

        if simulate_end_to_end:
            pipeline_on_simulated_data_end_to_end = Pipeline(
                outdir=os.path.join(outdir, "end_to_end", pipeline.outdir),
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
                precomputed_tree_dir=None,
                precomputed_maximum_parsimony_dir=None,
                use_cached=use_cached,
                num_epochs=pipeline.num_epochs,
                device=pipeline.device,
                center=pipeline.center,
                step_size=pipeline.step_size,
                n_steps=pipeline.n_steps,
                keep_outliers=pipeline.keep_outliers,
                max_height=pipeline.max_height,
                max_path_height=pipeline.max_path_height,
            )
            pipeline_on_simulated_data_end_to_end.run()
            logger.info(f"time_simulate_end_to_end:\n" f"{pipeline_on_simulated_data_end_to_end.get_times()}")

        if simulate_from_trees_wo_ancestral_states:
            pipeline_on_simulated_data_from_trees_wo_ancestral_states = Pipeline(
                outdir=os.path.join(outdir, "from_trees_wo_ancestral_states", pipeline.outdir),
                max_seqs=None,
                max_sites=None,
                armstrong_cutoff=None,
                rate_matrix=None,
                n_process=pipeline.n_process,
                expected_number_of_MSAs=pipeline.max_families,  # BC we only generated max_families MSAs!
                max_families=pipeline.max_families,
                a3m_dir=a3m_simulated_dir,
                pdb_dir=None,
                precomputed_contact_dir=contact_simulated_dir,
                precomputed_tree_dir=pipeline.tree_dir,
                precomputed_maximum_parsimony_dir=None,
                use_cached=use_cached,
                num_epochs=pipeline.num_epochs,
                device=pipeline.device,
                center=pipeline.center,
                step_size=pipeline.step_size,
                n_steps=pipeline.n_steps,
                keep_outliers=pipeline.keep_outliers,
                max_height=pipeline.max_height,
                max_path_height=pipeline.max_path_height,
            )
            pipeline_on_simulated_data_from_trees_wo_ancestral_states.run()
            logger.info(
                f"time_simulate_from_trees_wo_ancestral_states:\n"
                f"{pipeline_on_simulated_data_from_trees_wo_ancestral_states.get_times()}"
            )

        if simulate_from_trees_w_ancestral_states:
            pipeline_on_simulated_data_from_trees_w_ancestral_states = Pipeline(
                outdir=os.path.join(outdir, "from_trees_w_ancestral_states", pipeline.outdir),
                max_seqs=None,
                max_sites=None,
                armstrong_cutoff=None,
                rate_matrix=None,
                n_process=pipeline.n_process,
                expected_number_of_MSAs=pipeline.max_families,  # BC we only generated max_families MSAs!
                max_families=pipeline.max_families,
                a3m_dir=a3m_simulated_dir,
                pdb_dir=None,
                precomputed_contact_dir=contact_simulated_dir,
                precomputed_tree_dir=None,
                precomputed_maximum_parsimony_dir=ancestral_states_simulated_dir,
                use_cached=use_cached,
                num_epochs=pipeline.num_epochs,
                device=pipeline.device,
                center=pipeline.center,
                step_size=pipeline.step_size,
                n_steps=pipeline.n_steps,
                keep_outliers=pipeline.keep_outliers,
                max_height=pipeline.max_height,
                max_path_height=pipeline.max_path_height,
            )
            pipeline_on_simulated_data_from_trees_w_ancestral_states.run()
            logger.info(
                f"time_simulate_from_trees_w_ancestral_states:\n"
                f"{pipeline_on_simulated_data_from_trees_w_ancestral_states.get_times()}"
            )
