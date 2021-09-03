from logging import Filter
import os
import time
import numpy as np

from typing import List, Optional, Union
from src import maximum_parsimony

from src.phylogeny_generation import PhylogenyGenerator
from src.contact_generation import ContactGenerator
from src.maximum_parsimony import MaximumParsimonyReconstructor
from src.transition_extraction import TransitionExtractor
from src.co_transition_extraction import CoTransitionExtractor
from src.matrix_generation import MatrixGenerator
from src.ratelearn import RateMatrixLearner
from src.counting import JTT


class PipelineContextError(Exception):
    pass


class Pipeline:
    r"""
    A Pipeline estimates rate matrices from MSAs and structures.

    Given MSAs (.a3m files) and PDB structure files (.pdb), the Pipeline
    consumes all this data and estimates rate matrices. The Pipeline consists
    of several steps:
    - Contact calling.
    - Phylogeny generation.
    - Maximum parsimony reconstruction.
    - Transition extraction.
    - Filtering and counting of transitions.
    - Rate matrix estimation.
    As such, the Pipeline takes as argument hyperparameters concerning all
    these steps too, for example, the armstrong_cutoff used to call contacts.

    It is possible to skip steps of the pipeline by providing the ground truth
    values for those steps. This allows testing the pipeline in an end-to-end
    simulation.

    The Pipeline is initialized with __init__() and only run when called with
    the run() method.

    Because caching is important for performance, cached results found in outdir
    will be reused if possible (assuming use_cached=True is probided to __init__)
    Two pipelines can use the same outdir so long as their 'global context' is the
    same. The 'global context' of a pipeline is determined by the values of all these:
    - a3m_dir
    - pdb_dir
    - precomputed_contact_dir
    - precomputed_tree_dir
    - precomputed_maximum_parsimony_dir
    If you try to use the same outdir for pipelines with different context, a PipelineContextError
    will be raised. You must use a DIFFERENT outdir for the new pipeline. This is
    necessary because if not, caching would lead to disastrous bugs!
    One the other hand, please benefir from using the same outdir for pipelines with
    the same context. You will save the time of rebuilding trees, etc. An example
    of this is if you are trying to determine how the amount of data used (max_families)
    affects estimation error. You can create the exact same pipeline only varying the
    max_families parameter. Most of the computation will be re-used!

    Args:
        outdir: Directory where the estimated matrices will be found.
            All the intermediate data will also be written here.
        max_seqs: MSAs will be subsampled down to this number of sequences
            for the purpose of phylogeny generation with FastTree.
        max_sites: MSAs will be subsampled down to this number of sites
            for the purpose of phylogeny generation with FastTree.
        armstrong_cutoff: Contact threshold
        rate_matrix: What rate matrix to use in FastTree for the phylogeny
            reconstruction step.
        use_cached: If True, will do nothing for the output files that
            already exists, effectively re-using them.
        num_epochs: The number of epochs of first order optimization
            used to solve for the MLE.
        device: The device to use for pytorch optimization. Should be
            'cuda' or 'cpu'.
        center: Quantization grid center
        step_size: Quantization grid step size (geometric)
        n_steps: Number of grid points left and right of center (for a total
            of 2 * n_steps + 1 grid points)
        keep_outliers: What to do with points that are outside the grid. If
            False, they will be dropped. If True, they will be assigned
            to the corresponding closest endpoint of the grid.
        max_height: Use only transitions whose starting node is at height
            at most max_height from the leaves in its subtree. This is
            used to filter out unreliable maximum parsimony transitions.
        max_path_height: Use only transitions whose starting node is at height
            at most max_path_height from the leaves in its subtree, in terms
            of the NUMBER OF EDGES. This is used to filter out unreliable
            maximum parsimony transitions.
        n_process: How many processes to use.
        a3m_dir_full: The MSA directory which contains ALL the MSAs.
            This is needed because if one wants to run the pipeline
            on a subset of families (via max_families argument),
            then we need a way to determine what those families
            are. These are chosen uniformly randomly from the families
            in a3m_dir_full. Note that a3m_dir_full need not be the
            same as a3m_dir, for example when performing an end-to-end
            simulation: In this case, a3m_dir_full will point to the
            _real_ MSAs, while a3m_dir will point to the _synthetic_
            MSAs. Since subsampling a3m_dir uniformly at random
            is not guaranteed to produce the same families as
            subsampling from a3m_dir_full uniformly at random
            (given the same seed), then a3m_dir_full is needed.
        expected_number_of_MSAs: This is just used to check that the
            directory with the MSAs a3m_dir_full has the expected number
            of files. This is needed to make sure that when using
            the max_families feature to run the pipeline on a subset of
            families, the same families are chosen every single time.
        max_families: One can choose to run the pipeline on ONLY the
            first 'max_families'. This is super useful for testing the pipeline
            before running it on all data.
        a3m_dir: Directory where the MSAs (.a3m files) are found, for the
            purpose of phylogeny reconstruction, maximum parsimony
            reconstruction and finally frequency matrix construction.
        pdb_dir: Directory where the PDB (.pdb) structure files are found,
            for the purpose of contact matrix construction.
        precomputed_contact_dir: If this is supplied, the contact estimation
            step will be skipped and 'precomputed_contact_dir' will be used as
            the contact matrices.
        precomputed_tree_dir: If this is supplied, the tree estimation step
            will be skipped and 'precomputed_tree_dir' will be used as the
            phylogenies.
        precomputed_maximum_parsimony_dir: If this is suppled, the maximum
            parsimony reconstruction step will be skipped and
            'precomputed_maximum_parsimony_dir' will be used as the
            maximum parsimony reconstructions.
        learn_pairwise_model: If True, will learn a co-evolution model.
            This step is very slow in generating the training data, so
            it is a good idea to skip at first.
        edge_or_cherry: If "edge", edge transitions will be used. If "cherry",
            cherry transitions will be used instead. Note that "cherry"
            transitions do not depend on the maximum parsimony reconstruction!
        method: rate matrix estimation method, or list of rate matrix estimation
            methods. Possibilities: "MLE", "JTT", "JTT-IPW".
            (Note: cheap baseline will be run anyway.)
        init_jtt_ipw: if to initialize the MLE optimizer with the JTT-IPW
            estimate.

    Attributes:
        tree_dir: Where the estimated phylogenies lie
        contact_dir: Where the contact matrices lie
        maximum_parsimony_dir: Where the estimated phylogenies with maximum
            parsimony reconstructions lie.
        transitions_dir: Where the single-site transitions lie
        matrices_dir: Where the single-site transition frequency matrices lie.
        co_transitions_dir: Where the pair-of-site transitions lie
        co_matrices_dir: Where the pair-of-site frequency matrices lie.
        time_***: The time taken for each step of the pipeline.
    """

    def __init__(
        self,
        outdir: str,
        max_seqs: Optional[int],
        max_sites: Optional[int],
        armstrong_cutoff: Optional[str],
        rate_matrix: Optional[str],
        use_cached: bool,
        num_epochs: int,
        device: str,
        center: float,
        step_size: float,
        n_steps: int,
        max_height: float,
        max_path_height: int,
        keep_outliers: bool,
        n_process: int,
        a3m_dir_full: str,
        expected_number_of_MSAs: int,
        max_families: int,
        a3m_dir: str,
        pdb_dir: Optional[str],
        precomputed_contact_dir: Optional[str],
        precomputed_tree_dir: Optional[str],
        precomputed_maximum_parsimony_dir: Optional[str],
        edge_or_cherry: str = "edge",
        method: Union[str, List[str]] = "MLE",
        learn_pairwise_model: float = False,
        init_jtt_ipw: bool = False,
    ):
        method = method[:]
        if type(method) is str:
            method = [method]
        # Baselines are cheap so run anyway
        if "JTT" not in method:
            method.append("JTT")
        if "JTT-IPW" not in method:
            method.append("JTT-IPW")

        # Check input validity
        if precomputed_tree_dir is not None or precomputed_maximum_parsimony_dir is not None:
            if max_seqs is not None or max_sites is not None or rate_matrix is not None:
                raise ValueError("If trees are provided, FastTree parameters should all be None!")
        if precomputed_maximum_parsimony_dir is not None:
            if precomputed_tree_dir is not None:
                raise ValueError(
                    "If precomputed_maximum_parsimony_dir is provided, then there is no point in providing"
                    " precomputed_tree_dir"
                )
        if precomputed_contact_dir is not None:
            if armstrong_cutoff is not None or pdb_dir is not None:
                raise ValueError(
                    "If precomputed_contact_dir is provided, then there is no point in providing"
                    " armstrong_cutoff or pdb_dir."
                )
        if device not in ['cuda', 'cpu']:
            raise ValueError(f"device should be 'cuda' or 'cpu', {device} provided.")
        # Check that the global context is correct.
        global_context = str([a3m_dir_full, a3m_dir, pdb_dir, precomputed_contact_dir, precomputed_tree_dir, precomputed_maximum_parsimony_dir])
        global_context_filepath = os.path.join(outdir, 'global_context.txt')
        if os.path.exists(global_context_filepath):
            previous_global_context = open(global_context_filepath, "r").read()
            if global_context != previous_global_context:
                raise PipelineContextError(
                    f"Trying to run pipeline with outdir from a previous "
                    f"pipeline with a different context. Please use a different "
                    f"outdir. Previous context: {previous_global_context}. "
                    f"New context: {global_context}."
                    f"outdir = {outdir}")
        else:
            os.makedirs(outdir)
            with open(global_context_filepath, "w") as global_context_file:
                global_context_file.write(global_context)
            os.system(f"chmod 555 {global_context_filepath}")

        self.outdir = outdir
        self.max_seqs = max_seqs
        self.max_sites = max_sites
        self.armstrong_cutoff = armstrong_cutoff
        self.rate_matrix = rate_matrix
        self.n_process = n_process
        self.a3m_dir_full = a3m_dir_full
        self.expected_number_of_MSAs = expected_number_of_MSAs
        self.max_families = max_families
        self.a3m_dir = a3m_dir
        self.pdb_dir = pdb_dir
        self.precomputed_contact_dir = precomputed_contact_dir
        self.precomputed_tree_dir = precomputed_tree_dir
        self.precomputed_maximum_parsimony_dir = precomputed_maximum_parsimony_dir
        self.use_cached = use_cached
        self.num_epochs = num_epochs
        self.device = device
        self.center = center
        self.step_size = step_size
        self.n_steps = n_steps
        self.keep_outliers = keep_outliers
        self.max_height = max_height
        self.max_path_height = max_path_height
        rate_matrix_name = str(rate_matrix).split('/')[-1]
        self.rate_matrix_name = rate_matrix_name
        self.learn_pairwise_model = learn_pairwise_model
        self.edge_or_cherry = edge_or_cherry
        self.method = method
        self.init_jtt_ipw = init_jtt_ipw

        # Output data directories
        # Where the phylogenies will be stored
        tree_params = f"{max_seqs}_seqs_{max_sites}_sites_{rate_matrix_name}_RM"
        self.tree_dir = os.path.join(outdir, f"trees_{tree_params}")
        # Where the contacts will be stored
        contact_params = f"{armstrong_cutoff}_angstrom"
        self.contact_dir = os.path.join(outdir, f"contacts_{contact_params}")
        # Where the maximum parsimony reconstructions will be stored
        maximum_parsimony_params = tree_params
        self.maximum_parsimony_dir = os.path.join(outdir, f"maximum_parsimony_{maximum_parsimony_params}")
        # Where the transitions obtained from the maximum parsimony phylogenies will be stored
        transitions_params = maximum_parsimony_params
        self.transitions_dir = os.path.join(outdir, f"transitions_{transitions_params}")
        # Where the transition matrices obtained by quantizing transition edges will be stored
        cherry_str = "" if edge_or_cherry == "edge" else "_cherry"
        filter_params = f"{center}_center_{step_size}_step_size_{n_steps}_n_steps_{keep_outliers}_outliers_{max_height}_max_height_{max_path_height}_max_path_height{cherry_str}"
        matrices_params = f"{max_families}_families__{transitions_params}__{filter_params}"
        self.matrices_dir = os.path.join(outdir, f"matrices__{matrices_params}")
        # Where the co-transitions obtained from the maximum parsimony phylogenies will be stored
        co_transitions_params = f"{maximum_parsimony_params}_{contact_params}"
        self.co_transitions_dir = os.path.join(
            outdir,
            f"co_transitions_{co_transitions_params}",
        )
        # Where the co-transition matrices obtained by quantizing transition edges will be stored
        co_matrices_params = f"{max_families}_families__{co_transitions_params}__{filter_params}"
        self.co_matrices_dir = os.path.join(
            outdir,
            f"co_matrices__{co_matrices_params}",
        )
        str_init_jtt_ipw = "_init-JTT-IPW" if init_jtt_ipw else ""
        optimizer_params = f"{num_epochs}_epochs{str_init_jtt_ipw}"
        learnt_rate_matrix_params = f"{matrices_params}__{optimizer_params}"
        self.learnt_rate_matrix_dir = os.path.join(
            outdir,
            f"Q1__{learnt_rate_matrix_params}"
        )
        learnt_rate_matrix_JTT_params = matrices_params
        self.learnt_rate_matrix_dir_JTT = os.path.join(
            outdir,
            f"Q1_JTT__{learnt_rate_matrix_JTT_params}"
        )
        learnt_rate_matrix_JTT_IPW_params = matrices_params
        self.learnt_rate_matrix_dir_JTT_IPW = os.path.join(
            outdir,
            f"Q1_JTT-IPW__{learnt_rate_matrix_JTT_IPW_params}"
        )
        learnt_co_rate_matrix_params = f"{co_matrices_params}__{optimizer_params}"
        self.learnt_co_rate_matrix_dir = os.path.join(
            outdir,
            f"Q2__{learnt_co_rate_matrix_params}"
        )
        learnt_co_rate_matrix_JTT_params = co_matrices_params
        self.learnt_co_rate_matrix_dir_JTT = os.path.join(
            outdir,
            f"Q2_JTT__{learnt_co_rate_matrix_JTT_params}"
        )
        learnt_co_rate_matrix_JTT_IPW_params = co_matrices_params
        self.learnt_co_rate_matrix_dir_JTT_IPW = os.path.join(
            outdir,
            f"Q2_JTT-IPW__{learnt_co_rate_matrix_JTT_IPW_params}"
        )

    def run(self):
        max_seqs = self.max_seqs
        max_sites = self.max_sites
        armstrong_cutoff = self.armstrong_cutoff
        rate_matrix = self.rate_matrix
        use_cached = self.use_cached
        num_epochs = self.num_epochs
        device = self.device
        center = self.center
        step_size = self.step_size
        n_steps = self.n_steps
        max_height = self.max_height
        max_path_height = self.max_path_height
        keep_outliers = self.keep_outliers
        n_process = self.n_process
        a3m_dir_full = self.a3m_dir_full
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
        learnt_rate_matrix_dir = self.learnt_rate_matrix_dir
        learnt_rate_matrix_dir_JTT = self.learnt_rate_matrix_dir_JTT
        learnt_rate_matrix_dir_JTT_IPW = self.learnt_rate_matrix_dir_JTT_IPW
        learnt_co_rate_matrix_dir = self.learnt_co_rate_matrix_dir
        learnt_co_rate_matrix_dir_JTT = self.learnt_co_rate_matrix_dir_JTT
        learnt_co_rate_matrix_dir_JTT_IPW = self.learnt_co_rate_matrix_dir_JTT_IPW
        precomputed_contact_dir = self.precomputed_contact_dir
        precomputed_tree_dir = self.precomputed_tree_dir
        precomputed_maximum_parsimony_dir = self.precomputed_maximum_parsimony_dir
        learn_pairwise_model = self.learn_pairwise_model
        edge_or_cherry = self.edge_or_cherry
        method = self.method
        init_jtt_ipw = self.init_jtt_ipw
        path_mask_Q2 = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "../../input_data/synthetic_rate_matrices/mask_Q2.txt"
        )

        # First we need to generate the phylogenies
        t_start = time.time()
        if precomputed_tree_dir is None and precomputed_maximum_parsimony_dir is None:
            phylogeny_generator = PhylogenyGenerator(
                a3m_dir_full=a3m_dir_full,
                a3m_dir=a3m_dir,
                n_process=n_process,
                expected_number_of_MSAs=expected_number_of_MSAs,
                outdir=tree_dir,
                max_seqs=max_seqs,
                max_sites=max_sites,
                max_families=max_families,
                rate_matrix=rate_matrix,
                use_cached=use_cached,
            )
            phylogeny_generator.run()
        else:
            # Trees provided!
            if precomputed_tree_dir is None:
                # Case 1: We start from the trees w/ancestral states.
                assert precomputed_maximum_parsimony_dir is not None
            else:
                # Case 2: We start from the trees wo/ancestral states.
                assert precomputed_tree_dir is not None and precomputed_maximum_parsimony_dir is None
                tree_dir = precomputed_tree_dir
        self.time_PhylogenyGenerator = time.time() - t_start

        # Generate the contacts
        t_start = time.time()
        if precomputed_contact_dir is None:
            assert armstrong_cutoff is not None
            assert pdb_dir is not None
            contact_generator = ContactGenerator(
                a3m_dir_full=a3m_dir_full,
                a3m_dir=a3m_dir,
                pdb_dir=pdb_dir,
                armstrong_cutoff=armstrong_cutoff,
                n_process=n_process,
                expected_number_of_families=expected_number_of_MSAs,
                outdir=contact_dir,
                max_families=max_families,
                use_cached=use_cached,
            )
            contact_generator.run()
        else:
            assert armstrong_cutoff is None
            assert pdb_dir is None
            contact_dir = precomputed_contact_dir
        self.time_ContactGenerator = time.time() - t_start

        # Generate the maximum parsimony reconstructions
        t_start = time.time()
        if precomputed_maximum_parsimony_dir is None:
            maximum_parsimony_reconstructor = MaximumParsimonyReconstructor(
                a3m_dir_full=a3m_dir_full,
                a3m_dir=a3m_dir,
                tree_dir=tree_dir,
                n_process=n_process,
                expected_number_of_MSAs=expected_number_of_MSAs,
                outdir=maximum_parsimony_dir,
                max_families=max_families,
                use_cached=use_cached,
            )
            maximum_parsimony_reconstructor.run()
        else:
            assert precomputed_tree_dir is None
            maximum_parsimony_dir = precomputed_maximum_parsimony_dir
        self.time_MaximumParsimonyReconstructor = time.time() - t_start

        # Generate single-site transitions
        t_start = time.time()
        transition_extractor = TransitionExtractor(
            a3m_dir_full=a3m_dir_full,
            a3m_dir=a3m_dir,
            parsimony_dir=maximum_parsimony_dir,
            n_process=n_process,
            expected_number_of_MSAs=expected_number_of_MSAs,
            outdir=transitions_dir,
            max_families=max_families,
            use_cached=use_cached,
        )
        transition_extractor.run()
        self.time_TransitionExtractor = time.time() - t_start

        # Generate single-site transition matrices
        t_start = time.time()
        matrix_generator = MatrixGenerator(
            a3m_dir_full=a3m_dir_full,
            a3m_dir=a3m_dir,
            transitions_dir=transitions_dir,
            n_process=n_process,
            expected_number_of_MSAs=expected_number_of_MSAs,
            outdir=matrices_dir,
            max_families=max_families,
            num_sites=1,
            use_cached=use_cached,
            center=center,
            step_size=step_size,
            n_steps=n_steps,
            keep_outliers=keep_outliers,
            max_height=max_height,
            max_path_height=max_path_height,
            edge_or_cherry=edge_or_cherry,
        )
        matrix_generator.run()
        self.time_MatrixGenerator_1 = time.time() - t_start

        # Estimate single-site rate matrix Q1 with JTT counting
        t_start = time.time()
        if "JTT" in method:
            single_site_rate_matrix_learner = JTT(
                frequency_matrices=os.path.join(matrices_dir, "matrices_by_quantized_branch_length.txt"),
                output_dir=learnt_rate_matrix_dir_JTT,
                mask=None,
                use_cached=use_cached,
            )
            single_site_rate_matrix_learner.train()
        self.time_RateMatrixLearner_JTT_1 = time.time() - t_start

        # Estimate single-site rate matrix Q1 with JTT-IPW counting
        t_start = time.time()
        if "JTT-IPW" in method:
            single_site_rate_matrix_learner = JTT(
                frequency_matrices=os.path.join(matrices_dir, "matrices_by_quantized_branch_length.txt"),
                output_dir=learnt_rate_matrix_dir_JTT_IPW,
                mask=None,
                use_cached=use_cached,
                ipw=True,
            )
            single_site_rate_matrix_learner.train()
        self.time_RateMatrixLearner_JTT_IPW_1 = time.time() - t_start

        # Estimate single-site rate matrix Q1 with MLE (pytorch)
        t_start = time.time()
        if "MLE" in method:
            single_site_rate_matrix_learner = RateMatrixLearner(
                frequency_matrices=os.path.join(matrices_dir, "matrices_by_quantized_branch_length.txt"),
                output_dir=learnt_rate_matrix_dir,
                stationnary_distribution=None,
                mask=None,
                # frequency_matrices_sep=",",
                rate_matrix_parameterization="pande_reversible",
                device=device,
                use_cached=use_cached,
                initialization=self.get_learned_Q1_JTT_IPW() if init_jtt_ipw else None,
            )
            single_site_rate_matrix_learner.train(
                lr=1e-1,
                num_epochs=num_epochs,
                do_adam=True,
            )
        self.time_RateMatrixLearner_1 = time.time() - t_start

        # Generate co-transitions
        t_start = time.time()
        if learn_pairwise_model:
            co_transition_extractor = CoTransitionExtractor(
                a3m_dir_full=a3m_dir_full,
                a3m_dir=a3m_dir,
                parsimony_dir=maximum_parsimony_dir,
                n_process=n_process,
                expected_number_of_MSAs=expected_number_of_MSAs,
                outdir=co_transitions_dir,
                max_families=max_families,
                contact_dir=contact_dir,
                use_cached=use_cached,
            )
            co_transition_extractor.run()
        self.time_CoTransitionExtractor = time.time() - t_start

        # Generate co-transition matrices
        t_start = time.time()
        if learn_pairwise_model:
            matrix_generator_pairwise = MatrixGenerator(
                a3m_dir_full=a3m_dir_full,
                a3m_dir=a3m_dir,
                transitions_dir=co_transitions_dir,
                n_process=n_process,
                expected_number_of_MSAs=expected_number_of_MSAs,
                outdir=co_matrices_dir,
                max_families=max_families,
                num_sites=2,
                use_cached=use_cached,
                center=center,
                step_size=step_size,
                n_steps=n_steps,
                keep_outliers=keep_outliers,
                max_height=max_height,
                max_path_height=max_path_height,
                edge_or_cherry=edge_or_cherry,
            )
            matrix_generator_pairwise.run()
        self.time_MatrixGenerator_2 = time.time() - t_start

        # Estimate pair-of-sites rate matrix Q2 with JTT counting
        t_start = time.time()
        if learn_pairwise_model:
            if "JTT" in method:
                pair_of_site_rate_matrix_learner = JTT(
                    frequency_matrices=os.path.join(co_matrices_dir, "matrices_by_quantized_branch_length.txt"),
                    output_dir=learnt_co_rate_matrix_dir_JTT,
                    mask=path_mask_Q2,
                    use_cached=use_cached,
                )
                pair_of_site_rate_matrix_learner.train()
        self.time_RateMatrixLearner_JTT_2 = time.time() - t_start

        # Estimate pair-of-sites rate matrix Q2 with JTT-IPW counting
        t_start = time.time()
        if learn_pairwise_model:
            if "JTT-IPW" in method:
                pair_of_site_rate_matrix_learner = JTT(
                    frequency_matrices=os.path.join(co_matrices_dir, "matrices_by_quantized_branch_length.txt"),
                    output_dir=learnt_co_rate_matrix_dir_JTT_IPW,
                    mask=path_mask_Q2,
                    use_cached=use_cached,
                    ipw=True,
                )
                pair_of_site_rate_matrix_learner.train()
        self.time_RateMatrixLearner_JTT_IPW_2 = time.time() - t_start

        # Estimate pair-of-sites rate matrix Q2 with MLE (pytorch)
        t_start = time.time()
        if learn_pairwise_model:
            if "MLE" in method:
                pair_of_site_rate_matrix_learner = RateMatrixLearner(
                    frequency_matrices=os.path.join(co_matrices_dir, "matrices_by_quantized_branch_length.txt"),
                    output_dir=learnt_co_rate_matrix_dir,
                    stationnary_distribution=None,
                    mask=path_mask_Q2,
                    # frequency_matrices_sep=",",
                    rate_matrix_parameterization="pande_reversible",
                    device=device,
                    use_cached=use_cached,
                    initialization=self.get_learned_Q2_JTT_IPW() if init_jtt_ipw else None,
                )
                pair_of_site_rate_matrix_learner.train(
                    lr=1e-1,
                    num_epochs=num_epochs,
                    do_adam=True,
                )
        self.time_RateMatrixLearner_2 = time.time() - t_start

        self.time_total = (
            self.time_PhylogenyGenerator
            + self.time_ContactGenerator
            + self.time_MaximumParsimonyReconstructor
            + self.time_TransitionExtractor
            + self.time_MatrixGenerator_1
            + self.time_RateMatrixLearner_1
            + self.time_RateMatrixLearner_JTT_1
            + self.time_RateMatrixLearner_JTT_IPW_1
            + self.time_CoTransitionExtractor
            + self.time_MatrixGenerator_2
            + self.time_RateMatrixLearner_2
            + self.time_RateMatrixLearner_JTT_2
            + self.time_RateMatrixLearner_JTT_IPW_2
        )

    def get_times(self) -> str:
        r"""
        Returns a string message with the time taken for each part of the pipeline.
        Useful for profiling and finding bottlenecks.
        """
        res = (
            f"time_total = {self.time_total}. Breakdown:\n"
            + f"time_PhylogenyGenerator = {self.time_PhylogenyGenerator}\n"
            + f"time_ContactGenerator = {self.time_ContactGenerator}\n"
            + f"time_MaximumParsimonyReconstructor = {self.time_MaximumParsimonyReconstructor}\n"
            + f"time_TransitionExtractor = {self.time_TransitionExtractor}\n"
            + f"time_MatrixGenerator_1 = {self.time_MatrixGenerator_1}\n"
            + f"time_RateMatrixLearner_1 = {self.time_RateMatrixLearner_1}\n"
            + f"time_RateMatrixLearner_JTT_1 = {self.time_RateMatrixLearner_JTT_1}\n"
            + f"time_RateMatrixLearner_JTT_IPW_1 = {self.time_RateMatrixLearner_JTT_IPW_1}\n"
            + f"time_CoTransitionExtractor = {self.time_CoTransitionExtractor}\n"
            + f"time_MatrixGenerator_2 = {self.time_MatrixGenerator_2}\n"
            + f"time_RateMatrixLearner_2 = {self.time_RateMatrixLearner_2}\n"
            + f"time_RateMatrixLearner_JTT_2 = {self.time_RateMatrixLearner_JTT_2}\n"
            + f"time_RateMatrixLearner_JTT_IPW_2 = {self.time_RateMatrixLearner_JTT_IPW_2}\n"
        )
        return res

    def __str__(self):
        res = \
            "PIPELINE with:\n" \
            f"outdir = {self.outdir}\n" \
            f"max_seqs = {self.max_seqs}\n" \
            f"max_sites = {self.max_sites}\n" \
            f"armstrong_cutoff = {self.armstrong_cutoff}\n" \
            f"rate_matrix = {self.rate_matrix}\n" \
            f"use_cached = {self.use_cached}\n" \
            f"num_epochs = {self.num_epochs}\n" \
            f"device = {self.device}\n" \
            f"center = {self.center}\n" \
            f"step_size = {self.step_size}\n" \
            f"n_steps = {self.n_steps}\n" \
            f"max_height = {self.max_height}\n" \
            f"max_path_height = {self.max_path_height}\n" \
            f"edge_or_cherry = {self.edge_or_cherry}\n" \
            f"method = {self.method}\n" \
            f"keep_outliers = {self.keep_outliers}\n" \
            f"n_process = {self.n_process}\n" \
            f"a3m_dir_full = {self.a3m_dir_full}\n" \
            f"expected_number_of_MSAs = {self.expected_number_of_MSAs}\n" \
            f"max_families = {self.max_families}\n" \
            f"a3m_dir = {self.a3m_dir}\n" \
            f"pdb_dir = {self.pdb_dir}\n" \
            f"precomputed_contact_dir = {self.precomputed_contact_dir}\n" \
            f"precomputed_tree_dir = {self.precomputed_tree_dir}\n" \
            f"precomputed_maximum_parsimony_dir = {self.precomputed_maximum_parsimony_dir}\n" \
            f"learn_pairwise_model = {self.learn_pairwise_model}\n" \
            f"init_jtt_ipw = {self.init_jtt_ipw}\n"
        return res

    def get_learned_Q1(self) -> np.array:
        return np.loadtxt(
            os.path.join(
                self.learnt_rate_matrix_dir,
                "learned_matrix.txt",
            )
        )

    def get_learned_Q2(self) -> np.array:
        return np.loadtxt(
            os.path.join(
                self.learnt_co_rate_matrix_dir,
                "learned_matrix.txt",
            )
        )

    def get_learned_Q1_JTT(self) -> np.array:
        return np.loadtxt(
            os.path.join(
                self.learnt_rate_matrix_dir_JTT,
                "learned_matrix.txt",
            )
        )

    def get_learned_Q2_JTT(self) -> np.array:
        return np.loadtxt(
            os.path.join(
                self.learnt_co_rate_matrix_dir_JTT,
                "learned_matrix.txt",
            )
        )

    def get_learned_Q1_JTT_IPW(self) -> np.array:
        return np.loadtxt(
            os.path.join(
                self.learnt_rate_matrix_dir_JTT_IPW,
                "learned_matrix.txt",
            )
        )

    def get_learned_Q2_JTT_IPW(self) -> np.array:
        return np.loadtxt(
            os.path.join(
                self.learnt_co_rate_matrix_dir_JTT_IPW,
                "learned_matrix.txt",
            )
        )
