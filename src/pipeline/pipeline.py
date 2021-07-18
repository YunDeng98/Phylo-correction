import os
import time

from typing import Optional

from src.phylogeny_generation import PhylogenyGenerator
from src.contact_generation import ContactGenerator
from src.maximum_parsimony import MaximumParsimonyReconstructor
from src.transition_extraction import TransitionExtractor
from src.co_transition_extraction import CoTransitionExtractor
from src.matrix_generation import MatrixGenerator
from src.ratelearn import RateMatrixLearner


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
        n_process: How many processes to use.
        expected_number_of_MSAs: This is just used to check that the
            directory with the MSAs has the expected number of files.
            I.e. just used to perform a pedantic check.
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
        keep_outliers: bool,
        n_process: int,
        expected_number_of_MSAs: int,
        max_families: int,
        a3m_dir: str,
        pdb_dir: Optional[str],
        precomputed_contact_dir: Optional[str],
        precomputed_tree_dir: Optional[str],
        precomputed_maximum_parsimony_dir: Optional[str],
    ):
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
        self.precomputed_tree_dir = precomputed_tree_dir
        self.precomputed_maximum_parsimony_dir = precomputed_maximum_parsimony_dir
        self.use_cached = use_cached
        self.num_epochs = num_epochs
        self.device = device
        self.center = center
        self.step_size = step_size
        self.n_steps = n_steps
        self.keep_outliers = keep_outliers

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
        self.matrices_dir = os.path.join(outdir, f"matrices_{max_seqs}_seqs_{max_sites}_sites__{center}_center_{step_size}_step_size_{n_steps}_n_steps_{keep_outliers}_outliers")
        # Where the co-transitions obtained from the maximum parsimony phylogenies will be stored
        self.co_transitions_dir = os.path.join(
            outdir,
            f"co_transitions_{max_seqs}_seqs_{max_sites}_sites_{armstrong_cutoff}",
        )
        # Where the co-transition matrices obtained by quantizing transition edges will be stored
        self.co_matrices_dir = os.path.join(
            outdir,
            f"co_matrices_{max_seqs}_seqs_{max_sites}_sites_{armstrong_cutoff}__{center}_center_{step_size}_step_size_{n_steps}_n_steps_{keep_outliers}_outliers",
        )
        self.learnt_rate_matrix_dir = os.path.join(
            outdir,
            f"Q1_{max_seqs}_seqs_{max_sites}_sites__{center}_center_{step_size}_step_size_{n_steps}_n_steps_{keep_outliers}_outliers__{num_epochs}_epochs"
        )
        self.learnt_co_rate_matrix_dir = os.path.join(
            outdir,
            f"Q2_{max_seqs}_seqs_{max_sites}_sites_{armstrong_cutoff}__{center}_center_{step_size}_step_size_{n_steps}_n_steps_{keep_outliers}_outliers__{num_epochs}_epochs"
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
        keep_outliers = self.keep_outliers
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
        learnt_rate_matrix_dir = self.learnt_rate_matrix_dir
        learnt_co_rate_matrix_dir = self.learnt_co_rate_matrix_dir
        precomputed_contact_dir = self.precomputed_contact_dir
        precomputed_tree_dir = self.precomputed_tree_dir
        precomputed_maximum_parsimony_dir = self.precomputed_maximum_parsimony_dir

        # First we need to generate the phylogenies
        t_start = time.time()
        if precomputed_tree_dir is None and precomputed_maximum_parsimony_dir is None:
            phylogeny_generator = PhylogenyGenerator(
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
        )
        matrix_generator.run()
        self.time_MatrixGenerator_1 = time.time() - t_start

        # Generate co-transitions
        t_start = time.time()
        co_transition_extractor = CoTransitionExtractor(
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
        matrix_generator_pairwise = MatrixGenerator(
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
        )
        matrix_generator_pairwise.run()
        self.time_MatrixGenerator_2 = time.time() - t_start

        # Estimate single-site rate matrix Q1
        t_start = time.time()
        single_site_rate_matrix_learner = RateMatrixLearner(
            frequency_matrices=os.path.join(matrices_dir, "matrices_by_quantized_branch_length.txt"),
            output_dir=learnt_rate_matrix_dir,
            stationnary_distribution=None,
            mask=None,
            # frequency_matrices_sep=",",
            rate_matrix_parameterization="pande_reversible",
            device=device,
            use_cached=use_cached,
        )
        single_site_rate_matrix_learner.train(
            lr=1e-1,
            num_epochs=num_epochs,
            do_adam=True,
        )
        self.time_RateMatrixLearner_1 = time.time() - t_start

        # Estimate single-site rate matrix Q2
        t_start = time.time()
        pair_of_site_rate_matrix_learner = RateMatrixLearner(
            frequency_matrices=os.path.join(co_matrices_dir, "matrices_by_quantized_branch_length.txt"),
            output_dir=learnt_co_rate_matrix_dir,
            stationnary_distribution=None,
            mask=None,
            # frequency_matrices_sep=",",
            rate_matrix_parameterization="pande_reversible",
            device=device,
            use_cached=use_cached,
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
            + self.time_CoTransitionExtractor
            + self.time_MatrixGenerator_2
            + self.time_RateMatrixLearner_1
            + self.time_RateMatrixLearner_2
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
            + f"time_CoTransitionExtractor = {self.time_CoTransitionExtractor}\n"
            + f"time_MatrixGenerator_2 = {self.time_MatrixGenerator_2}\n"
            + f"time_RateMatrixLearner_1 = {self.time_RateMatrixLearner_1}\n"
        )
        return res
