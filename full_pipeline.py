import os
import sys
from typing import Optional

from phylogeny_generation import PhylogenyGenerator
from contact_generation import ContactGenerator
from maximum_parsimony import MaximumParsimonyReconstructor
from transition_extraction import TransitionExtractor
from co_transition_extraction import CoTransitionExtractor
from matrix_generation import MatrixGenerator
from simulation import Simulator

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

    fileHandler = logging.FileHandler("full_pipeline.log")
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)


class Pipeline():
    def __init__(
        self,
        outdir: str,
        max_seqs: int,
        max_sites: int,
        armstrong_cutoff: Optional[str],
        rate_matrix: str,
        n_process: int,
        expected_number_of_MSAs: int,
        max_families: int,
        a3m_dir: str,
        pdb_dir: Optional[str],
        precomputed_contact_dir: Optional[str] = None,
    ):
        self.outdir = outdir
        self.max_seqs = max_seqs
        self.max_sites = max_sites
        self.armstrong_cutoff = armstrong_cutoff
        self.rate_matrix = rate_matrix
        self.n_process = n_process
        self.expected_number_of_MSAs = expected_number_of_MSAs
        self.max_families = max_families
        self.a3m_dir = a3m_dir
        self.pdb_dir = pdb_dir
        self.precomputed_contact_dir = precomputed_contact_dir

        # Output data directories
        # Where the phylogenies will be stored
        self.tree_dir = os.path.join(outdir, f"trees_{max_seqs}_seqs_{max_sites}_sites")
        # Where the contacts will be stored
        self.contact_dir = os.path.join(outdir, f"contacts_{armstrong_cutoff}")
        # Where the maximum parsimony reconstructions will be stored
        self.maximum_parsimony_dir = os.path.join(outdir, f"maximum_parsimony_{max_seqs}_seqs_{max_sites}_sites")
        # Where the transitions obtained from the maximum parsimony phylogenies will be stored
        self.transitions_dir = os.path.join(outdir, f"transitions_{max_seqs}_seqs_{max_sites}_sites")
        # Where the transition matrices obtained by quantizing transition edges will be stored
        self.matrices_dir = os.path.join(outdir, f"matrices_{max_seqs}_seqs_{max_sites}_sites")
        # Where the co-transitions obtained from the maximum parsimony phylogenies will be stored
        self.co_transitions_dir = os.path.join(outdir, f"co_transitions_{max_seqs}_seqs_{max_sites}_sites_{armstrong_cutoff}")
        # Where the co-transition matrices obtained by quantizing transition edges will be stored
        self.co_matrices_dir = os.path.join(outdir, f"co_matrices_{max_seqs}_seqs_{max_sites}_sites_{armstrong_cutoff}")

    def run(self):
        max_seqs = self.max_seqs
        max_sites = self.max_sites
        armstrong_cutoff = self.armstrong_cutoff
        rate_matrix = self.rate_matrix
        n_process = self.n_process
        expected_number_of_MSAs = self.expected_number_of_MSAs
        max_families = self.max_families
        a3m_dir = self.a3m_dir
        pdb_dir = self.pdb_dir
        tree_dir = self.tree_dir
        contact_dir = self.contact_dir
        maximum_parsimony_dir = self.maximum_parsimony_dir
        transitions_dir = self.transitions_dir
        matrices_dir = self.matrices_dir
        co_transitions_dir = self.co_transitions_dir
        co_matrices_dir = self.co_matrices_dir
        precomputed_contact_dir = self.precomputed_contact_dir

        # First we need to generate the phylogenies
        phylogeny_generator = PhylogenyGenerator(
            a3m_dir=a3m_dir,
            n_process=n_process,
            expected_number_of_MSAs=expected_number_of_MSAs,
            outdir=tree_dir,
            max_seqs=max_seqs,
            max_sites=max_sites,
            max_families=max_families,
            rate_matrix=rate_matrix,
        )
        phylogeny_generator.run()

        # Generate the contacts
        if precomputed_contact_dir is None:
            assert(armstrong_cutoff is not None)
            assert(pdb_dir is not None)
            contact_generator = ContactGenerator(
                a3m_dir=a3m_dir,
                pdb_dir=pdb_dir,
                armstrong_cutoff=armstrong_cutoff,
                n_process=n_process,
                expected_number_of_families=expected_number_of_MSAs,
                outdir=contact_dir,
                max_families=max_families,
            )
            contact_generator.run()
        else:
            assert(armstrong_cutoff is None)
            assert(pdb_dir is None)
            contact_dir = precomputed_contact_dir

        # Generate the maximum parsimony reconstructions
        maximum_parsimony_reconstructor = MaximumParsimonyReconstructor(
            a3m_dir=a3m_dir,
            tree_dir=tree_dir,
            n_process=n_process,
            expected_number_of_MSAs=expected_number_of_MSAs,
            outdir=maximum_parsimony_dir,
            max_families=max_families,
        )
        maximum_parsimony_reconstructor.run()

        # Generate single-site transitions
        transition_extractor = TransitionExtractor(
            a3m_dir=a3m_dir,
            parsimony_dir=maximum_parsimony_dir,
            n_process=n_process,
            expected_number_of_MSAs=expected_number_of_MSAs,
            outdir=transitions_dir,
            max_families=max_families,
        )
        transition_extractor.run()

        # Generate single-site transition matrices
        matrix_generator = MatrixGenerator(
            a3m_dir=a3m_dir,
            transitions_dir=transitions_dir,
            n_process=n_process,
            expected_number_of_MSAs=expected_number_of_MSAs,
            outdir=matrices_dir,
            max_families=max_families,
            num_sites=1,
        )
        matrix_generator.run()

        # Generate co-transitions
        co_transition_extractor = CoTransitionExtractor(
            a3m_dir=a3m_dir,
            parsimony_dir=maximum_parsimony_dir,
            n_process=n_process,
            expected_number_of_MSAs=expected_number_of_MSAs,
            outdir=co_transitions_dir,
            max_families=max_families,
            contact_dir=contact_dir,
        )
        co_transition_extractor.run()

        # Generate co-transition matrices
        matrix_generator_pairwise = MatrixGenerator(
            a3m_dir=a3m_dir,
            transitions_dir=co_transitions_dir,
            n_process=n_process,
            expected_number_of_MSAs=expected_number_of_MSAs,
            outdir=co_matrices_dir,
            max_families=max_families,
            num_sites=2,
        )
        matrix_generator_pairwise.run()


class EndToEndSimulator:
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
        a3m_simulated_dir = os.path.join(outdir, 'a3m_simulated')
        contact_simulated_dir = os.path.join(outdir, 'contacts_simulated')
        ancestral_states_simulated_dir = os.path.join(outdir, 'ancestral_states_simulated')

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


def _end_to_end_simulator_test_minimal():
    pipeline = Pipeline(
        outdir='pipeline_output_test',
        max_seqs=1024,
        max_sites=1024,
        armstrong_cutoff=None,
        rate_matrix='None',
        n_process=3,
        expected_number_of_MSAs=3,
        max_families=3,
        a3m_dir='a3m_test',
        pdb_dir=None,
        precomputed_contact_dir=None,
    )

    end_to_end_simulator = EndToEndSimulator(
        outdir='test_outputs/_end_to_end_simulator_test_minimal',
        pipeline=pipeline,
        simulation_pct_interacting_positions=0.66,
        Q1_ground_truth='synthetic_rate_matrices/Q1_ground_truth.txt',
        Q2_ground_truth='synthetic_rate_matrices/Q2_ground_truth.txt',
        fast_tree_rate_matrix='synthetic_rate_matrices/Q1_uniform_FastTree.txt',
    )
    end_to_end_simulator.run()


def _end_to_end_simulator_test_uniform():
    pipeline = Pipeline(
        outdir='pipeline_output_test',
        max_seqs=1024,
        max_sites=1024,
        armstrong_cutoff=None,
        rate_matrix='None',
        n_process=3,
        expected_number_of_MSAs=3,
        max_families=3,
        a3m_dir='a3m_test',
        pdb_dir=None,
        precomputed_contact_dir=None,
    )

    end_to_end_simulator = EndToEndSimulator(
        outdir='test_outputs/_end_to_end_simulator_test_uniform',
        pipeline=pipeline,
        simulation_pct_interacting_positions=0.66,
        Q1_ground_truth='synthetic_rate_matrices/Q1_uniform.txt',
        Q2_ground_truth='synthetic_rate_matrices/Q2_uniform.txt',
        fast_tree_rate_matrix='synthetic_rate_matrices/Q1_uniform_FastTree.txt',
    )
    end_to_end_simulator.run()


def _end_to_end_simulator_test_uniform_constrained():
    pipeline = Pipeline(
        outdir='pipeline_output_test',
        max_seqs=1024,
        max_sites=1024,
        armstrong_cutoff=None,
        rate_matrix='None',
        n_process=3,
        expected_number_of_MSAs=3,
        max_families=3,
        a3m_dir='a3m_test',
        pdb_dir=None,
        precomputed_contact_dir=None,
    )

    end_to_end_simulator = EndToEndSimulator(
        outdir='test_outputs/_end_to_end_simulator_test_uniform_constrained',
        pipeline=pipeline,
        simulation_pct_interacting_positions=0.66,
        Q1_ground_truth='synthetic_rate_matrices/Q1_uniform.txt',
        Q2_ground_truth='synthetic_rate_matrices/Q2_uniform_constrained.txt',
        fast_tree_rate_matrix='synthetic_rate_matrices/Q1_uniform_FastTree.txt',
    )
    end_to_end_simulator.run()


def _end_to_end_simulator_test_large_matrices():
    pipeline = Pipeline(
        outdir='test_outputs/_end_to_end_simulator_test_large_matrices_pipeline_output',
        max_seqs=4,
        max_sites=4,
        armstrong_cutoff=9.0,
        rate_matrix='None',
        n_process=3,
        expected_number_of_MSAs=15051,
        max_families=3,
        a3m_dir='a3m',
        pdb_dir='pdb',
    )
    pipeline.run()

    end_to_end_simulator = EndToEndSimulator(
        outdir='test_outputs/_end_to_end_simulator_test_large_matrices',
        pipeline=pipeline,
        simulation_pct_interacting_positions=0.66,
        Q1_ground_truth='synthetic_rate_matrices/Q1_uniform.txt',
        Q2_ground_truth='synthetic_rate_matrices/Q2_uniform.txt',
        fast_tree_rate_matrix='synthetic_rate_matrices/Q1_uniform_FastTree.txt',
    )
    end_to_end_simulator.run()


def _end_to_end_simulator_test_real_data_small():
    r"""
    Takes 2 min.
    """
    pipeline = Pipeline(
        outdir='test_outputs/_end_to_end_simulator_test_real_data_small_pipeline_output',
        max_seqs=4,
        max_sites=4,
        armstrong_cutoff=8.0,
        rate_matrix='None',
        n_process=3,
        expected_number_of_MSAs=15051,
        max_families=3,
        a3m_dir='a3m',
        pdb_dir='pdb',
    )
    pipeline.run()

    end_to_end_simulator = EndToEndSimulator(
        outdir='test_outputs/_end_to_end_simulator_test_real_data_small_uniform',
        pipeline=pipeline,
        simulation_pct_interacting_positions=0.66,
        Q1_ground_truth='synthetic_rate_matrices/Q1_uniform.txt',
        Q2_ground_truth='synthetic_rate_matrices/Q2_uniform.txt',
        fast_tree_rate_matrix='synthetic_rate_matrices/Q1_uniform_FastTree.txt',
    )
    end_to_end_simulator.run()

    end_to_end_simulator = EndToEndSimulator(
        outdir='test_outputs/_end_to_end_simulator_test_real_data_small_uniform_constrained',
        pipeline=pipeline,
        simulation_pct_interacting_positions=0.66,
        Q1_ground_truth='synthetic_rate_matrices/Q1_uniform.txt',
        Q2_ground_truth='synthetic_rate_matrices/Q2_uniform_constrained.txt',
        fast_tree_rate_matrix='synthetic_rate_matrices/Q1_uniform_FastTree.txt',
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
        outdir='test_outputs/_end_to_end_simulator_test_real_data_mediumish_pipeline_output',
        max_seqs=4,
        max_sites=128,
        armstrong_cutoff=8.0,
        rate_matrix='None',
        n_process=3,
        expected_number_of_MSAs=15051,
        max_families=3,
        a3m_dir='a3m',
        pdb_dir='pdb',
    )
    pipeline.run()

    end_to_end_simulator = EndToEndSimulator(
        outdir='test_outputs/_end_to_end_simulator_test_real_data_mediumish_uniform',
        pipeline=pipeline,
        simulation_pct_interacting_positions=0.66,
        Q1_ground_truth='synthetic_rate_matrices/Q1_uniform.txt',
        Q2_ground_truth='synthetic_rate_matrices/Q2_uniform.txt',
        fast_tree_rate_matrix='synthetic_rate_matrices/Q1_uniform_FastTree.txt',
    )
    end_to_end_simulator.run()

    end_to_end_simulator = EndToEndSimulator(
        outdir='test_outputs/_end_to_end_simulator_test_real_data_mediumish_uniform_constrained',
        pipeline=pipeline,
        simulation_pct_interacting_positions=0.66,
        Q1_ground_truth='synthetic_rate_matrices/Q1_uniform.txt',
        Q2_ground_truth='synthetic_rate_matrices/Q2_uniform_constrained.txt',
        fast_tree_rate_matrix='synthetic_rate_matrices/Q1_uniform_FastTree.txt',
    )
    end_to_end_simulator.run()


def _end_to_end_simulator_test_real_data_medium():
    r"""
    Takes 10 min.
    """
    pipeline = Pipeline(
        outdir='test_outputs/_end_to_end_simulator_test_real_data_medium_pipeline_output',
        max_seqs=128,
        max_sites=128,
        armstrong_cutoff=8.0,
        rate_matrix='None',
        n_process=3,
        expected_number_of_MSAs=15051,
        max_families=3,
        a3m_dir='a3m',
        pdb_dir='pdb',
    )
    pipeline.run()

    end_to_end_simulator = EndToEndSimulator(
        outdir='test_outputs/_end_to_end_simulator_test_real_data_medium_uniform',
        pipeline=pipeline,
        simulation_pct_interacting_positions=0.66,
        Q1_ground_truth='synthetic_rate_matrices/Q1_uniform.txt',
        Q2_ground_truth='synthetic_rate_matrices/Q2_uniform.txt',
        fast_tree_rate_matrix='synthetic_rate_matrices/Q1_uniform_FastTree.txt',
    )
    end_to_end_simulator.run()

    end_to_end_simulator = EndToEndSimulator(
        outdir='test_outputs/_end_to_end_simulator_test_real_data_medium_uniform_constrained',
        pipeline=pipeline,
        simulation_pct_interacting_positions=0.66,
        Q1_ground_truth='synthetic_rate_matrices/Q1_uniform.txt',
        Q2_ground_truth='synthetic_rate_matrices/Q2_uniform_constrained.txt',
        fast_tree_rate_matrix='synthetic_rate_matrices/Q1_uniform_FastTree.txt',
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
        outdir='test_outputs/_test_fast_tree_pipeline_output',
        max_seqs=4,
        max_sites=1024,
        armstrong_cutoff=8.0,
        rate_matrix='None',
        n_process=3,
        expected_number_of_MSAs=15051,
        max_families=3,
        a3m_dir='a3m',
        pdb_dir='pdb',
    )
    pipeline.run()

    end_to_end_simulator = EndToEndSimulator(
        outdir='test_outputs/_test_fast_tree_uniform',
        pipeline=pipeline,
        simulation_pct_interacting_positions=0.0,
        Q1_ground_truth='synthetic_rate_matrices/Q1_uniform.txt',
        Q2_ground_truth='synthetic_rate_matrices/Q2_uniform.txt',    # Doesn't matter bc 0% interactions
        fast_tree_rate_matrix='synthetic_rate_matrices/Q1_uniform_FastTree.txt',
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
        outdir='test_outputs/_test_fast_tree_2_pipeline_output',
        max_seqs=4,
        max_sites=1024,
        armstrong_cutoff=8.0,
        rate_matrix='None',
        n_process=3,
        expected_number_of_MSAs=15051,
        max_families=3,
        a3m_dir='a3m',
        pdb_dir='pdb',
    )
    pipeline.run()

    end_to_end_simulator = EndToEndSimulator(
        outdir='test_outputs/_test_fast_tree_2_uniform',
        pipeline=pipeline,
        simulation_pct_interacting_positions=0.0,
        Q1_ground_truth='synthetic_rate_matrices/WAG_matrix.txt',
        Q2_ground_truth='synthetic_rate_matrices/Q2_uniform.txt',  # Doesn't matter bc 0% interactions
        fast_tree_rate_matrix='None',
    )
    end_to_end_simulator.run()


def _tests():
    _end_to_end_simulator_test_minimal()
    _end_to_end_simulator_test_uniform()
    _end_to_end_simulator_test_uniform_constrained()
    _end_to_end_simulator_test_large_matrices()
    _end_to_end_simulator_test_real_data_small()
    _end_to_end_simulator_test_real_data_mediumish()
    _end_to_end_simulator_test_real_data_medium()
    _test_fast_tree()
    _test_fast_tree_2()


def _main():
    init_logger()

    # _tests()

    # # Takes a loooooong time (days)
    # pipeline = Pipeline(
    #     outdir='real_outputs/pipeline_output',
    #     max_seqs=1024,
    #     max_sites=1024,
    #     armstrong_cutoff=8.0,
    #     rate_matrix='None',
    #     n_process=32,
    #     expected_number_of_MSAs=15051,
    #     max_families=15051,
    #     a3m_dir='a3m',
    #     pdb_dir='pdb',
    # )
    # pipeline.run()

    # For the end-to-end simulation, I'll first use only 3 families
    # to test it.
    pipeline = Pipeline(
        outdir='real_outputs/pipeline_output',
        max_seqs=1024,
        max_sites=1024,
        armstrong_cutoff=None,
        rate_matrix='None',
        n_process=3,
        expected_number_of_MSAs=15051,
        max_families=3,
        a3m_dir='a3m',
        pdb_dir=None,
    )
    end_to_end_simulator = EndToEndSimulator(
        outdir='real_outputs/end_to_end_simulation_with_independent_sites',
        pipeline=pipeline,
        simulation_pct_interacting_positions=0.0,
        Q1_ground_truth='synthetic_rate_matrices/WAG_matrix.txt',
        Q2_ground_truth='synthetic_rate_matrices/Q2_uniform.txt',
        fast_tree_rate_matrix='None',
    )
    end_to_end_simulator.run()

    # end_to_end_simulator = EndToEndSimulator(
    #     outdir='real_outputs/end_to_end_simulation_uniform',
    #     pipeline=pipeline,
    #     simulation_pct_interacting_positions=0.66,
    #     Q1_ground_truth='synthetic_rate_matrices/Q1_uniform.txt',
    #     Q2_ground_truth='synthetic_rate_matrices/Q2_uniform.txt',
    #     fast_tree_rate_matrix='synthetic_rate_matrices/Q1_uniform_FastTree.txt',
    # )
    # end_to_end_simulator.run()

    # end_to_end_simulator = EndToEndSimulator(
    #     outdir='test_outputs/_end_to_end_simulator_test_real_data_medium_uniform_constrained',
    #     pipeline=pipeline,
    #     simulation_pct_interacting_positions=0.66,
    #     Q1_ground_truth='synthetic_rate_matrices/Q1_uniform.txt',
    #     Q2_ground_truth='synthetic_rate_matrices/Q2_uniform_constrained.txt',
    #     fast_tree_rate_matrix='synthetic_rate_matrices/Q1_uniform_FastTree.txt',
    # )
    # end_to_end_simulator.run()


if __name__ == "__main__":
    _main()
