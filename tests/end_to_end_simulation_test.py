import unittest
import tempfile
from src.end_to_end_simulation import EndToEndSimulator
from collections import defaultdict
import numpy as np
import pytest
import os
from src.pipeline import Pipeline
import time


class TestEndToEndSimulation(unittest.TestCase):
    @pytest.mark.slow
    def test_shared_cache(self):
        """
        We want to be able to use the same output directory for all pipelines
        and all end-to-end simulations. I.e. we don't want to _ever_ have to
        run FastTree more than 2 x 15051 times! (once for the estimating the
        ground truth topologies, and once for the end-to-end simulations).

        To achieve this, the caching system must be working properly. If
        it is not working properly, a computation which should be performed
        is skipped and wrong results are loaded from the cache. This has
        disastrous consequences. On the other hand, if the caching system
        is working properly, it represents massive speedups in running
        experiments, specially large ones that can take days.

        This test therefore checks that the caching system is working correctly:
        we run different pipelines and end-to-end simulations in a world where
        the cache directory is the same for everybody, and compare the learnt
        rate matrices against a world where the cache directory is _different_
        for everybody. If the results agree, this is strong evidence that
        the caching system is working as expected.

        This test uses pipelines and end-to-end simulations that are similar
        in spirit to those we run on real data, except that they are performed
        on smaller protein families. Thus, they aim to cover realistic use cases
        to give us as much confidence as possible.
        """

        def run_experiment_and_return_learned_rate_matrices(
            ### Pipeline parameters
            pipeline_outdir=None,
            max_seqs=8,  # 1024,
            max_sites=16,  # 1024,
            armstrong_cutoff=8.0,
            rate_matrix='input_data/synthetic_rate_matrices/WAG_FastTree.txt',
            use_cached=True,
            num_epochs=10,  # 200,
            device='cpu',
            center=1.0,  # 0.06,
            step_size=0.5,  # 0.1
            n_steps=1,  # 50,
            max_height=1000.0,
            max_path_height=1000,
            keep_outliers=True,  # False
            n_process=32,
            expected_number_of_MSAs=32,
            max_families=32,  # 15051
            a3m_dir_full=None,
            a3m_dir='test_input_data/a3m_32_families',  # 'input_data/a3m',
            pdb_dir='test_input_data/pdb_32_families',  # 'input_data/pdb',
            precomputed_contact_dir=None,
            precomputed_tree_dir=None,
            precomputed_maximum_parsimony_dir=None,
            edge_or_cherry="edge",
            method="MLE",
            learn_pairwise_model=False,
            init_jtt_ipw=False,
            rate_matrix_parameterization="pande_reversible",
            ### End-to-end simulator parameters
            end_to_end_simulator_outdir=None,
            # pipeline=pipeline,
            simulation_pct_interacting_positions=0.0,
            Q1_ground_truth="input_data/synthetic_rate_matrices/WAG_matrix.txt",
            Q2_ground_truth="input_data/synthetic_rate_matrices/Q2_uniform_constrained.txt",
            fast_tree_rate_matrix='input_data/synthetic_rate_matrices/WAG_FastTree.txt',
            simulate_end_to_end=True,
            simulate_from_trees_wo_ancestral_states=True,
            simulate_from_trees_w_ancestral_states=True,
            # use_cached=use_cached,
        ):
            learned_rate_matrices = defaultdict(dict)
            pipeline = Pipeline(
                outdir=pipeline_outdir,
                max_seqs=max_seqs,
                max_sites=max_sites,
                armstrong_cutoff=armstrong_cutoff,
                rate_matrix=rate_matrix,
                use_cached=use_cached,
                num_epochs=num_epochs,
                device=device,
                center=center,
                step_size=step_size,
                n_steps=n_steps,
                max_height=max_height,
                max_path_height=max_path_height,
                keep_outliers=keep_outliers,
                n_process=n_process,
                a3m_dir_full=a3m_dir_full,
                expected_number_of_MSAs=expected_number_of_MSAs,
                max_families=max_families,
                a3m_dir=a3m_dir,
                pdb_dir=pdb_dir,
                precomputed_contact_dir=precomputed_contact_dir,
                precomputed_tree_dir=precomputed_tree_dir,
                precomputed_maximum_parsimony_dir=precomputed_maximum_parsimony_dir,
                edge_or_cherry=edge_or_cherry,
                method=method,
                learn_pairwise_model=learn_pairwise_model,
                init_jtt_ipw=init_jtt_ipw,
                rate_matrix_parameterization=rate_matrix_parameterization,
            )
            pipeline.run()
            learned_rate_matrices['real_data'] = pipeline.get_learned_Q1()

            end_to_end_simulator = EndToEndSimulator(
                outdir=end_to_end_simulator_outdir,
                pipeline=pipeline,
                simulation_pct_interacting_positions=simulation_pct_interacting_positions,
                Q1_ground_truth=Q1_ground_truth,
                Q2_ground_truth=Q2_ground_truth,
                fast_tree_rate_matrix=fast_tree_rate_matrix,
                simulate_end_to_end=simulate_end_to_end,
                simulate_from_trees_wo_ancestral_states=simulate_from_trees_wo_ancestral_states,
                simulate_from_trees_w_ancestral_states=simulate_from_trees_w_ancestral_states,
                use_cached=use_cached,
            )
            end_to_end_simulator.run()
            learned_rate_matrices['end_to_end'] = end_to_end_simulator.pipeline_on_simulated_data_end_to_end.get_learned_Q1()
            learned_rate_matrices['wo_ancestral_states'] = end_to_end_simulator.pipeline_on_simulated_data_from_trees_wo_ancestral_states.get_learned_Q1()
            learned_rate_matrices['w_ancestral_states'] = end_to_end_simulator.pipeline_on_simulated_data_from_trees_w_ancestral_states.get_learned_Q1()
            return learned_rate_matrices

        max_seqs_list = [8, 9]
        max_families_list = [32, 1]
        experiment_list = ['experiment_1', 'experiment_2', 'experiment_3']
        times = []  # Just to see how long each experiment takes.

        def learned_rate_matrices_aux(increment: int):
            """
            Calling with increment=0 leads to all pipelines using pipeline_0 as
            the output directory and all end-to-end simulators using
            end_to_end_simulator_0 as the output directory.

            Calling with increment=1 leads to pipeline i using pipeline_{i+1} as
            the output directory and end-to-end simulator i using
            end_to_end_simulator_{i+1} as the output directory.
            """
            key = increment
            randomized_max_families_list = max_families_list[:] if increment == 0 else max_families_list[::-1]
            with tempfile.TemporaryDirectory() as root_dir:
                learned_rate_matrices_for_max_seqs = {}
                for max_seqs in max_seqs_list:
                    learned_rate_matrices_for_families = {}
                    for max_families in randomized_max_families_list:
                        curr_learned_rate_matrices = {}
                        # First experiment: Using all transitions and no filtering
                        st = time.time()
                        curr_learned_rate_matrices[experiment_list[0]] = run_experiment_and_return_learned_rate_matrices(
                            pipeline_outdir=os.path.join(root_dir, f"pipeline_{key}"),
                            end_to_end_simulator_outdir=os.path.join(root_dir, f"end_to_end_simulator_{key}"),
                            max_seqs=max_seqs,
                            max_families=max_families,
                        )
                        times.append(time.time() - st)
                        key += increment
                        # Second experiment: Using transition filtering.
                        st = time.time()
                        curr_learned_rate_matrices[experiment_list[1]] = run_experiment_and_return_learned_rate_matrices(
                            pipeline_outdir=os.path.join(root_dir, f"pipeline_{key}"),
                            end_to_end_simulator_outdir=os.path.join(root_dir, f"end_to_end_simulator_{key}"),
                            max_seqs=max_seqs,
                            max_families=max_families,
                            max_path_height=1,
                            max_height=3.0,
                        )
                        times.append(time.time() - st)
                        key += increment
                        # Third experiment: Using cherries
                        st = time.time()
                        curr_learned_rate_matrices[experiment_list[2]] = run_experiment_and_return_learned_rate_matrices(
                            pipeline_outdir=os.path.join(root_dir, f"pipeline_{key}"),
                            end_to_end_simulator_outdir=os.path.join(root_dir, f"end_to_end_simulator_{key}"),
                            max_seqs=max_seqs,
                            max_families=max_families,
                            edge_or_cherry="cherry",
                            init_jtt_ipw=True,
                        )
                        times.append(time.time() - st)
                        key += increment
                        learned_rate_matrices_for_families[max_families] = curr_learned_rate_matrices
                    learned_rate_matrices_for_max_seqs[max_seqs] = learned_rate_matrices_for_families
            return learned_rate_matrices_for_max_seqs

        # Universe #1: Shared Caches
        learned_rate_matrices_shared_caches = learned_rate_matrices_aux(0)

        # Universe #2: Disjoint Caches
        learned_rate_matrices_disjoint_caches = learned_rate_matrices_aux(1)

        learned_rate_matrices_shared_caches_list = []
        learned_rate_matrices_disjoint_caches_list = []

        for max_seqs in max_seqs_list:
            for max_families in max_families_list:
                for experiment in experiment_list:
                    for what in ['real_data', 'w_ancestral_states', 'wo_ancestral_states', 'end_to_end']:
                        learned_rate_matrices_shared_caches_list.append(
                            learned_rate_matrices_shared_caches[max_seqs][max_families][experiment][what]
                        )
                        learned_rate_matrices_disjoint_caches_list.append(
                            learned_rate_matrices_disjoint_caches[max_seqs][max_families][experiment][what]
                        )
                        try:
                            np.testing.assert_almost_equal(
                                learned_rate_matrices_shared_caches[max_seqs][max_families][experiment][what],
                                learned_rate_matrices_disjoint_caches[max_seqs][max_families][experiment][what],
                            )
                        except:
                            raise ValueError(
                                f"Failed for \n"
                                f"max_families = {max_families}\n"
                                f"experiment = {experiment}\n"
                                f"what = {what}"
                            )

        # This is a duplicate of the above, but just in case
        for i in range(len(learned_rate_matrices_shared_caches_list)):
            error = np.sum(np.abs(learned_rate_matrices_shared_caches_list[i] - learned_rate_matrices_disjoint_caches_list[i]))
            assert(error < 1e-8)

        # Check that all rate matrices are different (because the previous check passes if
        # there is a bug where all the rate matrices are the same, i.e. when the parameters
        # don't modulate anything!)
        errors = []
        for what in [
            learned_rate_matrices_shared_caches_list,
            learned_rate_matrices_disjoint_caches_list
        ]:
            for i in range(len(what)):
                for j in range(i + 1, len(what), 1):
                    error = np.sum(np.abs(what[i] - what[j]))
                    errors.append(error)
        # Only 8 errors can should be exactly zero: those for the
        # cherry experiment w and wo ancestral states.
        errors = sorted(errors)
        print(f"errors = {errors}")
        for error in errors[8:]:
            assert(error > 0.1)
        for error in errors[:8]:
            assert(error < 0.1)

        # For identifying and debugging slow experiments.
        # print(f"times = {times}")
        # assert(False)

    @pytest.mark.slow
    def test_shared_cache_XRATE(self):
        """
        Same as above but for XRATE.
        """

        def run_experiment_and_return_learned_rate_matrices(
            ### Pipeline parameters
            pipeline_outdir=None,
            max_seqs=8,  # 1024,
            max_sites=16,  # 1024,
            armstrong_cutoff=8.0,
            rate_matrix='input_data/synthetic_rate_matrices/WAG_FastTree.txt',
            use_cached=True,
            num_epochs=10,  # 200,
            device='cpu',
            center=1.0,  # 0.06,
            step_size=0.5,  # 0.1
            n_steps=1,  # 50,
            max_height=1000.0,
            max_path_height=1000,
            keep_outliers=True,  # False
            n_process=32,
            expected_number_of_MSAs=32,
            max_families=32,  # 15051
            a3m_dir_full=None,
            a3m_dir='test_input_data/a3m_32_families',  # 'input_data/a3m',
            pdb_dir='test_input_data/pdb_32_families',  # 'input_data/pdb',
            precomputed_contact_dir=None,
            precomputed_tree_dir=None,
            precomputed_maximum_parsimony_dir=None,
            edge_or_cherry="edge",
            method="MLE",
            learn_pairwise_model=False,
            init_jtt_ipw=False,
            rate_matrix_parameterization="pande_reversible",
            xrate_grammar=None,
            ### End-to-end simulator parameters
            end_to_end_simulator_outdir=None,
            # pipeline=pipeline,
            simulation_pct_interacting_positions=0.0,
            Q1_ground_truth="input_data/synthetic_rate_matrices/WAG_matrix.txt",
            Q2_ground_truth="input_data/synthetic_rate_matrices/Q2_uniform_constrained.txt",
            fast_tree_rate_matrix='input_data/synthetic_rate_matrices/WAG_FastTree.txt',
            simulate_end_to_end=True,
            simulate_from_trees_wo_ancestral_states=True,
            simulate_from_trees_w_ancestral_states=True,
            # use_cached=use_cached,
        ):
            learned_rate_matrices = defaultdict(dict)
            pipeline = Pipeline(
                outdir=pipeline_outdir,
                max_seqs=max_seqs,
                max_sites=max_sites,
                armstrong_cutoff=armstrong_cutoff,
                rate_matrix=rate_matrix,
                use_cached=use_cached,
                num_epochs=num_epochs,
                device=device,
                center=center,
                step_size=step_size,
                n_steps=n_steps,
                max_height=max_height,
                max_path_height=max_path_height,
                keep_outliers=keep_outliers,
                n_process=n_process,
                a3m_dir_full=a3m_dir_full,
                expected_number_of_MSAs=expected_number_of_MSAs,
                max_families=max_families,
                a3m_dir=a3m_dir,
                pdb_dir=pdb_dir,
                precomputed_contact_dir=precomputed_contact_dir,
                precomputed_tree_dir=precomputed_tree_dir,
                precomputed_maximum_parsimony_dir=precomputed_maximum_parsimony_dir,
                edge_or_cherry=edge_or_cherry,
                method=method,
                learn_pairwise_model=learn_pairwise_model,
                init_jtt_ipw=init_jtt_ipw,
                rate_matrix_parameterization=rate_matrix_parameterization,
                xrate_grammar=xrate_grammar,
            )
            pipeline.run()
            learned_rate_matrices['real_data'] = pipeline.get_learned_Q1_XRATE()

            end_to_end_simulator = EndToEndSimulator(
                outdir=end_to_end_simulator_outdir,
                pipeline=pipeline,
                simulation_pct_interacting_positions=simulation_pct_interacting_positions,
                Q1_ground_truth=Q1_ground_truth,
                Q2_ground_truth=Q2_ground_truth,
                fast_tree_rate_matrix=fast_tree_rate_matrix,
                simulate_end_to_end=simulate_end_to_end,
                simulate_from_trees_wo_ancestral_states=simulate_from_trees_wo_ancestral_states,
                simulate_from_trees_w_ancestral_states=simulate_from_trees_w_ancestral_states,
                use_cached=use_cached,
            )
            end_to_end_simulator.run()
            learned_rate_matrices['end_to_end'] = end_to_end_simulator.pipeline_on_simulated_data_end_to_end.get_learned_Q1_XRATE()
            learned_rate_matrices['wo_ancestral_states'] = end_to_end_simulator.pipeline_on_simulated_data_from_trees_wo_ancestral_states.get_learned_Q1_XRATE()
            learned_rate_matrices['w_ancestral_states'] = end_to_end_simulator.pipeline_on_simulated_data_from_trees_w_ancestral_states.get_learned_Q1_XRATE()
            return learned_rate_matrices

        max_seqs_list = [8, 9]
        max_families_list = [2, 1]
        experiment_list = ['experiment_1', 'experiment_2', 'experiment_3']
        times = []  # Just to see how long each experiment takes.

        def learned_rate_matrices_aux(increment: int):
            """
            Calling with increment=0 leads to all pipelines using pipeline_0 as
            the output directory and all end-to-end simulators using
            end_to_end_simulator_0 as the output directory.

            Calling with increment=1 leads to pipeline i using pipeline_{i+1} as
            the output directory and end-to-end simulator i using
            end_to_end_simulator_{i+1} as the output directory.
            """
            key = increment
            randomized_max_families_list = max_families_list[:] if increment == 0 else max_families_list[::-1]
            with tempfile.TemporaryDirectory() as root_dir:
                learned_rate_matrices_for_max_seqs = {}
                for max_seqs in max_seqs_list:
                    learned_rate_matrices_for_families = {}
                    for max_families in randomized_max_families_list:
                        curr_learned_rate_matrices = {}
                        # First experiment: Using nullprot.eg
                        st = time.time()
                        curr_learned_rate_matrices[experiment_list[0]] = run_experiment_and_return_learned_rate_matrices(
                            pipeline_outdir=os.path.join(root_dir, f"pipeline_{key}"),
                            end_to_end_simulator_outdir=os.path.join(root_dir, f"end_to_end_simulator_{key}"),
                            max_seqs=max_seqs,
                            max_families=max_families,
                            method=["JTT-IPW", "XRATE"],
                            xrate_grammar="test_input_data/xrate_grammars/nullprot.eg",
                        )
                        times.append(time.time() - st)
                        key += increment
                        # Second experiment: Using WAG.eg
                        st = time.time()
                        curr_learned_rate_matrices[experiment_list[1]] = run_experiment_and_return_learned_rate_matrices(
                            pipeline_outdir=os.path.join(root_dir, f"pipeline_{key}"),
                            end_to_end_simulator_outdir=os.path.join(root_dir, f"end_to_end_simulator_{key}"),
                            max_seqs=max_seqs,
                            max_families=max_families,
                            method=["JTT-IPW", "XRATE"],
                            xrate_grammar="test_input_data/xrate_grammars/WAG.eg",
                        )
                        times.append(time.time() - st)
                        key += increment
                        # Third experiment: Using EQU.eg
                        st = time.time()
                        curr_learned_rate_matrices[experiment_list[2]] = run_experiment_and_return_learned_rate_matrices(
                            pipeline_outdir=os.path.join(root_dir, f"pipeline_{key}"),
                            end_to_end_simulator_outdir=os.path.join(root_dir, f"end_to_end_simulator_{key}"),
                            max_seqs=max_seqs,
                            max_families=max_families,
                            method=["JTT-IPW", "XRATE"],
                            xrate_grammar="test_input_data/xrate_grammars/EQU.eg",
                        )
                        times.append(time.time() - st)
                        key += increment
                        learned_rate_matrices_for_families[max_families] = curr_learned_rate_matrices
                    learned_rate_matrices_for_max_seqs[max_seqs] = learned_rate_matrices_for_families
            return learned_rate_matrices_for_max_seqs

        # Universe #1: Shared Caches
        learned_rate_matrices_shared_caches = learned_rate_matrices_aux(0)

        # Universe #2: Disjoint Caches
        learned_rate_matrices_disjoint_caches = learned_rate_matrices_aux(1)

        learned_rate_matrices_shared_caches_list = []
        learned_rate_matrices_disjoint_caches_list = []

        for max_seqs in max_seqs_list:
            for max_families in max_families_list:
                for experiment in experiment_list:
                    for what in ['real_data', 'w_ancestral_states', 'wo_ancestral_states', 'end_to_end']:
                        learned_rate_matrices_shared_caches_list.append(
                            learned_rate_matrices_shared_caches[max_seqs][max_families][experiment][what]
                        )
                        learned_rate_matrices_disjoint_caches_list.append(
                            learned_rate_matrices_disjoint_caches[max_seqs][max_families][experiment][what]
                        )
                        try:
                            np.testing.assert_almost_equal(
                                learned_rate_matrices_shared_caches[max_seqs][max_families][experiment][what],
                                learned_rate_matrices_disjoint_caches[max_seqs][max_families][experiment][what],
                            )
                        except:
                            raise ValueError(
                                f"Failed for \n"
                                f"max_families = {max_families}\n"
                                f"experiment = {experiment}\n"
                                f"what = {what}"
                            )

        # This is a duplicate of the above, but just in case
        for i in range(len(learned_rate_matrices_shared_caches_list)):
            error = np.sum(np.abs(learned_rate_matrices_shared_caches_list[i] - learned_rate_matrices_disjoint_caches_list[i]))
            assert(error < 1e-8)

        # Check that all rate matrices are different (because the previous check passes if
        # there is a bug where all the rate matrices are the same, i.e. when the parameters
        # don't modulate anything!)
        errors = []
        for what in [
            learned_rate_matrices_shared_caches_list,
            learned_rate_matrices_disjoint_caches_list
        ]:
            for i in range(len(what)):
                for j in range(i + 1, len(what), 1):
                    error = np.sum(np.abs(what[i] - what[j]))
                    errors.append(error)
        # Only 24 errors can should be exactly zero: those for the
        # experiment w and wo ancestral states.
        errors = sorted(errors)
        print(f"errors = {errors}")
        for error in errors[24:]:
            assert(error > 0.1)
        for error in errors[:24]:
            assert(error < 0.1)

        # For identifying and debugging slow experiments.
        # print(f"times = {times}")
        # assert(False)
