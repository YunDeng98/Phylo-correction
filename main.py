import sys

from src.pipeline import Pipeline
from src.end_to_end_simulation import EndToEndSimulator

import logging


def init_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # fmt_str = "[%(asctime)s] %(levelname)s @ line %(lineno)d: %(message)s"
    fmt_str = "[%(asctime)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt_str)

    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)

    fileHandler = logging.FileHandler("Phylo-correction.log")
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)


def test_end_to_end_simulation_real_data():
    r"""
    Takes 2 min.
    """
    pipeline = Pipeline(
        outdir="test_outputs/test_end_to_end_simulation_real_data_pipeline_output",
        max_seqs=4,
        max_sites=128,
        armstrong_cutoff=8.0,
        rate_matrix="None",
        n_process=3,
        expected_number_of_MSAs=15051,
        max_families=3,
        a3m_dir="input_data/a3m",
        pdb_dir="input_data/pdb",
    )
    pipeline.run()

    end_to_end_simulator = EndToEndSimulator(
        outdir="test_outputs/test_end_to_end_simulation_real_data_simulation_output",
        pipeline=pipeline,
        simulation_pct_interacting_positions=0.33,
        Q1_ground_truth="input_data/synthetic_rate_matrices/WAG_matrix.txt",
        Q2_ground_truth="input_data/synthetic_rate_matrices/Q2_uniform.txt",
        fast_tree_rate_matrix='None',
        simulate_end_to_end=True,
        simulate_from_trees_wo_ancestral_states=True,
        simulate_from_trees_w_ancestral_states=True,
    )
    end_to_end_simulator.run()


def _main():
    init_logger()

    test_end_to_end_simulation_real_data()

    # # Takes a loooooong time (days)
    # pipeline = Pipeline(
    #     outdir='output_data/pipeline_output',
    #     max_seqs=1024,
    #     max_sites=1024,
    #     armstrong_cutoff=8.0,
    #     rate_matrix='None',
    #     n_process=32,
    #     expected_number_of_MSAs=15051,
    #     max_families=15051,
    #     a3m_dir='input_data/a3m',
    #     pdb_dir='input_data/pdb',
    # )
    # pipeline.run()

    # # For the end-to-end simulation, I'll first use only 3 families
    # # to test it.
    # pipeline = Pipeline(
    #     outdir='output_data/pipeline_output',
    #     max_seqs=1024,
    #     max_sites=1024,
    #     armstrong_cutoff=None,
    #     rate_matrix='None',
    #     n_process=3,
    #     expected_number_of_MSAs=15051,
    #     max_families=3,
    #     a3m_dir='input_data/a3m',
    #     pdb_dir=None,
    # )
    # end_to_end_simulator = EndToEndSimulator(
    #     outdir='output_data/end_to_end_simulation_with_independent_sites',
    #     pipeline=pipeline,
    #     simulation_pct_interacting_positions=0.0,
    #     Q1_ground_truth='input_data/synthetic_rate_matrices/WAG_matrix.txt',
    #     Q2_ground_truth='input_data/synthetic_rate_matrices/Q2_uniform.txt',
    #     fast_tree_rate_matrix='None',
    # )
    # end_to_end_simulator.run()

    # # end_to_end_simulator = EndToEndSimulator(
    # #     outdir='output_data/end_to_end_simulation_uniform',
    # #     pipeline=pipeline,
    # #     simulation_pct_interacting_positions=0.66,
    # #     Q1_ground_truth='input_data/synthetic_rate_matrices/Q1_uniform.txt',
    # #     Q2_ground_truth='input_data/synthetic_rate_matrices/Q2_uniform.txt',
    # #     fast_tree_rate_matrix='input_data/synthetic_rate_matrices/Q1_uniform_FastTree.txt',
    # # )
    # # end_to_end_simulator.run()

    # # end_to_end_simulator = EndToEndSimulator(
    # #     outdir='test_outputs/_end_to_end_simulator_test_real_data_medium_uniform_constrained',
    # #     pipeline=pipeline,
    # #     simulation_pct_interacting_positions=0.66,
    # #     Q1_ground_truth='input_data/synthetic_rate_matrices/Q1_uniform.txt',
    # #     Q2_ground_truth='input_data/synthetic_rate_matrices/Q2_uniform_constrained.txt',
    # #     fast_tree_rate_matrix='input_data/synthetic_rate_matrices/Q1_uniform_FastTree.txt',
    # # )
    # # end_to_end_simulator.run()


if __name__ == "__main__":
    _main()
