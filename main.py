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


def _end_to_end_simulator_test_minimal():
    pipeline = Pipeline(
        outdir="pipeline_output_test",
        max_seqs=1024,
        max_sites=1024,
        armstrong_cutoff=None,
        rate_matrix="None",
        n_process=3,
        expected_number_of_MSAs=3,
        max_families=3,
        a3m_dir="input_data/a3m_test",
        pdb_dir=None,
        precomputed_contact_dir=None,
    )

    end_to_end_simulator = EndToEndSimulator(
        outdir="test_outputs/_end_to_end_simulator_test_minimal",
        pipeline=pipeline,
        simulation_pct_interacting_positions=0.66,
        Q1_ground_truth="input_data/synthetic_rate_matrices/Q1_ground_truth.txt",
        Q2_ground_truth="input_data/synthetic_rate_matrices/Q2_ground_truth.txt",
        fast_tree_rate_matrix="input_data/synthetic_rate_matrices/Q1_uniform_FastTree.txt",
        simulate_end_to_end=True,
        simulate_from_trees_wo_ancestral_states=True,
        simulate_from_trees_w_ancestral_states=True,
    )
    end_to_end_simulator.run()


def _end_to_end_simulator_test_uniform():
    pipeline = Pipeline(
        outdir="pipeline_output_test",
        max_seqs=1024,
        max_sites=1024,
        armstrong_cutoff=None,
        rate_matrix="None",
        n_process=3,
        expected_number_of_MSAs=3,
        max_families=3,
        a3m_dir="input_data/a3m_test",
        pdb_dir=None,
        precomputed_contact_dir=None,
    )

    end_to_end_simulator = EndToEndSimulator(
        outdir="test_outputs/_end_to_end_simulator_test_uniform",
        pipeline=pipeline,
        simulation_pct_interacting_positions=0.66,
        Q1_ground_truth="input_data/synthetic_rate_matrices/Q1_uniform.txt",
        Q2_ground_truth="input_data/synthetic_rate_matrices/Q2_uniform.txt",
        fast_tree_rate_matrix="input_data/synthetic_rate_matrices/Q1_uniform_FastTree.txt",
        simulate_end_to_end=True,
    )
    end_to_end_simulator.run()


def _end_to_end_simulator_test_uniform_constrained():
    pipeline = Pipeline(
        outdir="pipeline_output_test",
        max_seqs=1024,
        max_sites=1024,
        armstrong_cutoff=None,
        rate_matrix="None",
        n_process=3,
        expected_number_of_MSAs=3,
        max_families=3,
        a3m_dir="input_data/a3m_test",
        pdb_dir=None,
        precomputed_contact_dir=None,
    )

    end_to_end_simulator = EndToEndSimulator(
        outdir="test_outputs/_end_to_end_simulator_test_uniform_constrained",
        pipeline=pipeline,
        simulation_pct_interacting_positions=0.66,
        Q1_ground_truth="input_data/synthetic_rate_matrices/Q1_uniform.txt",
        Q2_ground_truth="input_data/synthetic_rate_matrices/Q2_uniform_constrained.txt",
        fast_tree_rate_matrix="input_data/synthetic_rate_matrices/Q1_uniform_FastTree.txt",
        simulate_end_to_end=True,
    )
    end_to_end_simulator.run()


def _end_to_end_simulator_test_large_matrices():
    pipeline = Pipeline(
        outdir="test_outputs/_end_to_end_simulator_test_large_matrices_pipeline_output",
        max_seqs=4,
        max_sites=4,
        armstrong_cutoff=9.0,
        rate_matrix="None",
        n_process=3,
        expected_number_of_MSAs=15051,
        max_families=3,
        a3m_dir="input_data/a3m",
        pdb_dir="input_data/pdb",
    )
    pipeline.run()

    end_to_end_simulator = EndToEndSimulator(
        outdir="test_outputs/_end_to_end_simulator_test_large_matrices",
        pipeline=pipeline,
        simulation_pct_interacting_positions=0.66,
        Q1_ground_truth="input_data/synthetic_rate_matrices/Q1_uniform.txt",
        Q2_ground_truth="input_data/synthetic_rate_matrices/Q2_uniform.txt",
        fast_tree_rate_matrix="input_data/synthetic_rate_matrices/Q1_uniform_FastTree.txt",
        simulate_end_to_end=True,
    )
    end_to_end_simulator.run()


def _end_to_end_simulator_test_real_data_small():
    r"""
    Takes 2 min.
    """
    pipeline = Pipeline(
        outdir="test_outputs/_end_to_end_simulator_test_real_data_small_pipeline_output",
        max_seqs=4,
        max_sites=4,
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
        outdir="test_outputs/_end_to_end_simulator_test_real_data_small_uniform",
        pipeline=pipeline,
        simulation_pct_interacting_positions=0.66,
        Q1_ground_truth="input_data/synthetic_rate_matrices/Q1_uniform.txt",
        Q2_ground_truth="input_data/synthetic_rate_matrices/Q2_uniform.txt",
        fast_tree_rate_matrix="input_data/synthetic_rate_matrices/Q1_uniform_FastTree.txt",
        simulate_end_to_end=True,
    )
    end_to_end_simulator.run()

    end_to_end_simulator = EndToEndSimulator(
        outdir="test_outputs/_end_to_end_simulator_test_real_data_small_uniform_constrained",
        pipeline=pipeline,
        simulation_pct_interacting_positions=0.66,
        Q1_ground_truth="input_data/synthetic_rate_matrices/Q1_uniform.txt",
        Q2_ground_truth="input_data/synthetic_rate_matrices/Q2_uniform_constrained.txt",
        fast_tree_rate_matrix="input_data/synthetic_rate_matrices/Q1_uniform_FastTree.txt",
        simulate_end_to_end=True,
    )
    end_to_end_simulator.run()


def _end_to_end_simulator_test_real_data_mediumish():
    r"""
    Takes 3:30 min.

    I've used this to check that FastTree is working by comparing
    the trees in the pipeline output (used for simulating MSAs)
    vs the ones obtained after running the full pipeline on the
    simulated data. This is easy because there are only 4 leaves
    in the trees.
    """
    pipeline = Pipeline(
        outdir="test_outputs/_end_to_end_simulator_test_real_data_mediumish_pipeline_output",
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
        outdir="test_outputs/_end_to_end_simulator_test_real_data_mediumish_uniform",
        pipeline=pipeline,
        simulation_pct_interacting_positions=0.66,
        Q1_ground_truth="input_data/synthetic_rate_matrices/Q1_uniform.txt",
        Q2_ground_truth="input_data/synthetic_rate_matrices/Q2_uniform.txt",
        fast_tree_rate_matrix="input_data/synthetic_rate_matrices/Q1_uniform_FastTree.txt",
        simulate_end_to_end=True,
    )
    end_to_end_simulator.run()

    end_to_end_simulator = EndToEndSimulator(
        outdir="test_outputs/_end_to_end_simulator_test_real_data_mediumish_uniform_constrained",
        pipeline=pipeline,
        simulation_pct_interacting_positions=0.66,
        Q1_ground_truth="input_data/synthetic_rate_matrices/Q1_uniform.txt",
        Q2_ground_truth="input_data/synthetic_rate_matrices/Q2_uniform_constrained.txt",
        fast_tree_rate_matrix="input_data/synthetic_rate_matrices/Q1_uniform_FastTree.txt",
        simulate_end_to_end=True,
    )
    end_to_end_simulator.run()


def _end_to_end_simulator_test_real_data_medium():
    r"""
    Takes 10 min.
    """
    pipeline = Pipeline(
        outdir="test_outputs/_end_to_end_simulator_test_real_data_medium_pipeline_output",
        max_seqs=128,
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
        outdir="test_outputs/_end_to_end_simulator_test_real_data_medium_uniform",
        pipeline=pipeline,
        simulation_pct_interacting_positions=0.66,
        Q1_ground_truth="input_data/synthetic_rate_matrices/Q1_uniform.txt",
        Q2_ground_truth="input_data/synthetic_rate_matrices/Q2_uniform.txt",
        fast_tree_rate_matrix="input_data/synthetic_rate_matrices/Q1_uniform_FastTree.txt",
        simulate_end_to_end=True,
    )
    end_to_end_simulator.run()

    end_to_end_simulator = EndToEndSimulator(
        outdir="test_outputs/_end_to_end_simulator_test_real_data_medium_uniform_constrained",
        pipeline=pipeline,
        simulation_pct_interacting_positions=0.66,
        Q1_ground_truth="input_data/synthetic_rate_matrices/Q1_uniform.txt",
        Q2_ground_truth="input_data/synthetic_rate_matrices/Q2_uniform_constrained.txt",
        fast_tree_rate_matrix="input_data/synthetic_rate_matrices/Q1_uniform_FastTree.txt",
        simulate_end_to_end=True,
    )
    end_to_end_simulator.run()


def _test_fast_tree():
    r"""
    Takes 2 min.

    I use it to test FastTree: There are only 4 leaves,
    and there is no co-evolution. I use FastTree to infer
    data with the same single-site matrix as was used to generate the data.
    """
    pipeline = Pipeline(
        outdir="test_outputs/_test_fast_tree_pipeline_output",
        max_seqs=4,
        max_sites=1024,
        armstrong_cutoff=8.0,
        rate_matrix="None",
        n_process=1,
        expected_number_of_MSAs=15051,
        max_families=1,
        a3m_dir="input_data/a3m",
        pdb_dir="input_data/pdb",
    )
    pipeline.run()

    end_to_end_simulator = EndToEndSimulator(
        outdir="test_outputs/_test_fast_tree_uniform",
        pipeline=pipeline,
        simulation_pct_interacting_positions=0.0,
        Q1_ground_truth="input_data/synthetic_rate_matrices/Q1_uniform.txt",
        Q2_ground_truth="input_data/synthetic_rate_matrices/Q2_uniform.txt",  # Doesn't matter bc 0% interactions
        fast_tree_rate_matrix="input_data/synthetic_rate_matrices/Q1_uniform_FastTree.txt",
        simulate_end_to_end=True,
    )
    end_to_end_simulator.run()


def _test_fast_tree_2():
    r"""
    Takes 1 min.

    I use it to test FastTree: There are only 4 leaves,
    and there is no co-evolution. I use FastTree to infer
    data with the same single-site matrix as was used to generate the data.
    """
    pipeline = Pipeline(
        outdir="test_outputs/_test_fast_tree_2_pipeline_output",
        max_seqs=4,
        max_sites=1024,
        armstrong_cutoff=8.0,
        rate_matrix="None",
        n_process=1,
        expected_number_of_MSAs=15051,
        max_families=1,
        a3m_dir="input_data/a3m",
        pdb_dir="input_data/pdb",
    )
    pipeline.run()

    end_to_end_simulator = EndToEndSimulator(
        outdir="test_outputs/_test_fast_tree_2_uniform",
        pipeline=pipeline,
        simulation_pct_interacting_positions=0.0,
        Q1_ground_truth="input_data/synthetic_rate_matrices/WAG_matrix.txt",
        Q2_ground_truth="input_data/synthetic_rate_matrices/Q2_uniform.txt",  # Doesn't matter bc 0% interactions
        fast_tree_rate_matrix="None",
        simulate_end_to_end=True,
    )
    end_to_end_simulator.run()


def _tests():
    _test_fast_tree()
    _test_fast_tree_2()
    _end_to_end_simulator_test_minimal()
    _end_to_end_simulator_test_uniform()
    _end_to_end_simulator_test_uniform_constrained()
    _end_to_end_simulator_test_large_matrices()
    _end_to_end_simulator_test_real_data_small()
    _end_to_end_simulator_test_real_data_mediumish()
    _end_to_end_simulator_test_real_data_medium()


def _main():
    init_logger()

    _tests()

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
