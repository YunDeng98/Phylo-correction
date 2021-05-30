import os

from typing import Optional

from src.phylogeny_generation import PhylogenyGenerator
from src.contact_generation import ContactGenerator
from src.maximum_parsimony import MaximumParsimonyReconstructor
from src.transition_extraction import TransitionExtractor
from src.co_transition_extraction import CoTransitionExtractor
from src.matrix_generation import MatrixGenerator


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
        n_process: How many processes to use.
        expected_number_of_MSAs: This is just used to check that the
            directory with the MSAs has the expected number of files.
            I.e. just used to perform a pedantic check.
        max_families: One can choose to run the pipeline on ONLY the
            first 'max_families'. This is super useful for testing the pipeline
            before running it on all data.
        a3m_dir: Directory where the MSAs (.a3m files) are found, for the
            purpose of phylogeny reconstruction, maximum parsomony
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
    """

    def __init__(
        self,
        outdir: str,
        max_seqs: Optional[int],
        max_sites: Optional[int],
        armstrong_cutoff: Optional[str],
        rate_matrix: Optional[str],
        n_process: int,
        expected_number_of_MSAs: int,
        max_families: int,
        a3m_dir: str,
        pdb_dir: Optional[str],
        precomputed_contact_dir: Optional[str] = None,
        precomputed_tree_dir: Optional[str] = None,
        precomputed_maximum_parsimony_dir: Optional[str] = None,
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
        self.co_transitions_dir = os.path.join(
            outdir,
            f"co_transitions_{max_seqs}_seqs_{max_sites}_sites_{armstrong_cutoff}",
        )
        # Where the co-transition matrices obtained by quantizing transition edges will be stored
        self.co_matrices_dir = os.path.join(
            outdir,
            f"co_matrices_{max_seqs}_seqs_{max_sites}_sites_{armstrong_cutoff}",
        )

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
        precomputed_tree_dir = self.precomputed_tree_dir
        precomputed_maximum_parsimony_dir = self.precomputed_maximum_parsimony_dir

        # First we need to generate the phylogenies
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

        # Generate the contacts
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
            )
            contact_generator.run()
        else:
            assert armstrong_cutoff is None
            assert pdb_dir is None
            contact_dir = precomputed_contact_dir

        # Generate the maximum parsimony reconstructions
        if precomputed_maximum_parsimony_dir is None:
            maximum_parsimony_reconstructor = MaximumParsimonyReconstructor(
                a3m_dir=a3m_dir,
                tree_dir=tree_dir,
                n_process=n_process,
                expected_number_of_MSAs=expected_number_of_MSAs,
                outdir=maximum_parsimony_dir,
                max_families=max_families,
            )
            maximum_parsimony_reconstructor.run()
        else:
            assert precomputed_tree_dir is None
            maximum_parsimony_dir = precomputed_maximum_parsimony_dir

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
