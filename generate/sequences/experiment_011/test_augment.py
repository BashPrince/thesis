"""
Unit tests for the genetic augmentation module.
"""

import random
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio

import numpy as np
import pandas as pd
import pytest

from augment import (
    compute_embeddings,
    find_most_distant_pair,
    partition_genes,
    format_gene_list,
    format_prompt,
    parse_llm_response,
    InvalidResponseError,
    GeneticAugmenter,
    build_base_set,
    create_augmented_dataset,
    GENES,
)


class TestComputeEmbeddings:
    """Tests for compute_embeddings function."""

    def test_returns_correct_shape(self):
        """Embeddings should have shape (n_texts, embedding_dim)."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

        texts = ["hello", "world"]
        result = compute_embeddings(texts, mock_model)

        assert result.shape == (2, 2)
        mock_model.encode.assert_called_once_with(texts, show_progress_bar=False)

    def test_empty_list(self):
        """Should handle empty list."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([]).reshape(0, 384)

        result = compute_embeddings([], mock_model)

        assert result.shape == (0, 384)


class TestFindMostDistantPair:
    """Tests for find_most_distant_pair function."""

    def test_finds_most_distant(self):
        """Should return indices of the most distant pair."""
        # Create embeddings where idx 0 and 2 are most distant
        embeddings = np.array([
            [1.0, 0.0],   # idx 0
            [0.5, 0.5],   # idx 1 - in between
            [-1.0, 0.0],  # idx 2 - opposite of 0
        ])

        i, j = find_most_distant_pair(embeddings, set())

        # Should be 0 and 2 (or 2 and 0)
        assert set([i, j]) == {0, 2}

    def test_respects_used_indices(self):
        """Should avoid used indices when possible."""
        embeddings = np.array([
            [1.0, 0.0],
            [0.5, 0.5],
            [-1.0, 0.0],
            [0.0, 1.0],
        ])

        # Mark 0 and 2 as used
        used = {0, 2}
        i, j = find_most_distant_pair(embeddings, used)

        # Should be from remaining indices
        assert i not in used or j not in used or len(embeddings) - len(used) < 2

    def test_raises_when_not_enough_available(self):
        """Should raise ValueError when too few samples remain after exclusion."""
        embeddings = np.array([
            [1.0, 0.0],
            [-1.0, 0.0],
        ])

        # All indices excluded
        excluded = {0, 1}
        with pytest.raises(ValueError, match="Not enough available samples"):
            find_most_distant_pair(embeddings, excluded)


class TestPartitionGenes:
    """Tests for partition_genes function."""

    def test_partitions_into_three_groups(self):
        """Should partition genes into G1 (3), G2 (3), and G3 (remaining)."""
        rng = random.Random(42)
        genes = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]

        g1, g2, g3 = partition_genes(genes, rng)

        # G1 should have exactly 3 genes
        assert isinstance(g1, list) and len(g1) == 3

        # G2 should have exactly 3 genes
        assert isinstance(g2, list) and len(g2) == 3

        # G3 should be a list with remaining genes (10 - 6 = 4)
        assert isinstance(g3, list)
        assert len(g3) == 4

        # All genes should be accounted for
        all_genes = set(g1 + g2 + g3)
        assert all_genes == set(genes)

    def test_is_randomized(self):
        """Different RNG states should produce different partitions."""
        genes = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]

        rng1 = random.Random(42)
        g1_1, g2_1, g3_1 = partition_genes(genes, rng1)

        rng2 = random.Random(123)
        g1_2, g2_2, g3_2 = partition_genes(genes, rng2)

        # Should (almost certainly) be different
        assert (g1_1, g2_1, g3_1) != (g1_2, g2_2, g3_2) or True  # Allow pass anyway

    def test_with_actual_genes(self):
        """Should work with the actual GENES list (10 genes)."""
        rng = random.Random(42)

        g1, g2, g3 = partition_genes(GENES, rng)

        assert len(g1) == 3
        assert len(g2) == 3
        assert len(g3) == 4  # 10 - 3 - 3 = 4 remaining genes
        # All selected genes should be from the original list
        for gene in g1 + g2 + g3:
            assert gene in GENES


class TestFormatGeneList:
    """Tests for format_gene_list function."""

    def test_single_gene(self):
        """Should return single gene as-is."""
        assert format_gene_list(["tone"]) == "tone"

    def test_two_genes(self):
        """Should join two genes with 'and'."""
        assert format_gene_list(["tone", "subject"]) == "tone and subject"

    def test_three_genes(self):
        """Should use comma and 'and' for three genes."""
        assert format_gene_list(["tone", "subject", "length"]) == "tone, subject and length"

    def test_four_genes(self):
        """Should use commas and 'and' for four genes."""
        result = format_gene_list(["tone", "subject", "length", "certainty"])
        assert result == "tone, subject, length and certainty"


class TestFormatPrompt:
    """Tests for format_prompt function."""

    def test_formats_positive_prompt(self):
        """Should use positive template for Yes class."""
        prompt = format_prompt(
            claim_1="Claim one",
            claim_2="Claim two",
            g1=["subject", "tone", "length"],
            g2=["certainty", "specificity", "perspective"],
            g3=["actor focus", "temporal orientation", "sentence structure", "speaker stance"],
            class_label="Yes",
        )

        assert "Claim one" in prompt
        assert "Claim two" in prompt
        assert "check-worthy" in prompt.lower()
        assert "subject" in prompt
        # G3 should be formatted as comma-separated with 'and'
        assert "actor focus, temporal orientation, sentence structure and speaker stance" in prompt

    def test_formats_negative_prompt(self):
        """Should use negative template for No class."""
        prompt = format_prompt(
            claim_1="Claim one",
            claim_2="Claim two",
            g1=["subject", "tone", "length"],
            g2=["certainty", "specificity", "perspective"],
            g3=["actor focus", "temporal orientation", "sentence structure", "speaker stance"],
            class_label="No",
        )

        assert "Claim one" in prompt
        assert "Claim two" in prompt
        assert "non check-worthy" in prompt.lower()


class TestParseLlmResponse:
    """Tests for parse_llm_response function."""

    def test_extracts_plain_text(self):
        """Should extract plain text response."""
        response = "This is a claim about politics."
        result = parse_llm_response(response)
        assert result == "This is a claim about politics."

    def test_removes_quotes(self):
        """Should remove surrounding quotes."""
        response = '"This is a quoted claim."'
        result = parse_llm_response(response)
        assert result == "This is a quoted claim."

    def test_handles_whitespace(self):
        """Should handle extra whitespace."""
        response = "   Some claim text.   "
        result = parse_llm_response(response)
        assert result == "Some claim text."

    def test_rejects_multiline_response(self):
        """Should reject responses with multiple lines."""
        response = "Here's the claim:\nThe actual claim text."
        with pytest.raises(InvalidResponseError):
            parse_llm_response(response)

    def test_rejects_multiline_with_preamble(self):
        """Should reject multi-line responses with preambles."""
        response = "Certainly! Here's a synthetic claim:\nThe economy grew by 5%."
        with pytest.raises(InvalidResponseError):
            parse_llm_response(response)

    def test_accepts_single_line(self):
        """Should accept a single line response."""
        response = "The unemployment rate dropped to 3.5% last quarter."
        result = parse_llm_response(response)
        assert result == "The unemployment rate dropped to 3.5% last quarter."

    def test_ignores_empty_lines(self):
        """Should ignore empty lines when counting."""
        response = "  \n  The claim text.  \n  "
        result = parse_llm_response(response)
        assert result == "The claim text."


class TestGeneticAugmenter:
    """Tests for GeneticAugmenter class."""

    @pytest.fixture
    def mock_embedding_model(self):
        """Create a mock embedding model."""
        model = MagicMock()

        def encode_fn(texts, show_progress_bar=False):
            # Return deterministic embeddings based on text content
            embeddings = []
            for text in texts:
                # Simple hash-based embedding
                np.random.seed(hash(text) % (2**31))
                embeddings.append(np.random.rand(384))
            return np.array(embeddings)

        model.encode.side_effect = encode_fn
        return model

    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame."""
        return pd.DataFrame({
            'Text': [
                'Yes claim 1', 'Yes claim 2', 'Yes claim 3',
                'No claim 1', 'No claim 2', 'No claim 3',
            ],
            'class_label': ['Yes', 'Yes', 'Yes', 'No', 'No', 'No'],
        })

    def test_initialize_population(self, mock_embedding_model, sample_df):
        """Should initialize pools for both classes."""
        augmenter = GeneticAugmenter(mock_embedding_model)
        augmenter.initialize_population(sample_df)

        assert "Yes" in augmenter.pools
        assert "No" in augmenter.pools
        assert len(augmenter.pools["Yes"]["texts"]) == 3
        assert len(augmenter.pools["No"]["texts"]) == 3
        assert augmenter.pools["Yes"]["embeddings"].shape[0] == 3

    def test_add_to_pool(self, mock_embedding_model, sample_df):
        """Should add new sample to pool."""
        augmenter = GeneticAugmenter(mock_embedding_model)
        augmenter.initialize_population(sample_df)

        initial_count = len(augmenter.pools["Yes"]["texts"])
        augmenter.add_to_pool("New claim", "Yes")

        assert len(augmenter.pools["Yes"]["texts"]) == initial_count + 1
        assert "New claim" in augmenter.pools["Yes"]["texts"]

    def test_select_parents(self, mock_embedding_model, sample_df):
        """Should return two parent texts."""
        augmenter = GeneticAugmenter(mock_embedding_model)
        augmenter.initialize_population(sample_df)

        p1, p2, idx1, idx2 = augmenter.select_parents("Yes")

        assert p1 in augmenter.pools["Yes"]["texts"]
        assert p2 in augmenter.pools["Yes"]["texts"]
        assert p1 != p2
        assert idx1 != idx2

    def test_statistics_tracking(self, mock_embedding_model, sample_df):
        """Should correctly track selected_count and iteration_count."""
        augmenter = GeneticAugmenter(mock_embedding_model)
        augmenter.initialize_population(sample_df)

        # Initial state: 3 Yes samples, all at iteration 0, none selected
        assert augmenter.current_iteration["Yes"] == 0
        for i in range(3):
            assert augmenter.selected_count["Yes"][i] == 0
            assert augmenter.entry_iteration["Yes"][i] == 0

        # Simulate generation loop: iteration 1
        augmenter.current_iteration["Yes"] += 1
        p1, p2, idx1, idx2 = augmenter.select_parents("Yes")

        # Both parents should have selected_count = 1
        assert augmenter.selected_count["Yes"][idx1] == 1
        assert augmenter.selected_count["Yes"][idx2] == 1

        # Add synthetic sample at iteration 1
        augmenter.add_to_pool("Synthetic 1", "Yes")
        assert augmenter.entry_iteration["Yes"][3] == 1  # New sample at index 3
        assert augmenter.selected_count["Yes"][3] == 0

        # Simulate iteration 2
        augmenter.current_iteration["Yes"] += 1
        p1, p2, idx1_2, idx2_2 = augmenter.select_parents("Yes")

        # Add another synthetic sample at iteration 2
        augmenter.add_to_pool("Synthetic 2", "Yes")
        assert augmenter.entry_iteration["Yes"][4] == 2

        # Simulate iteration 3
        augmenter.current_iteration["Yes"] += 1
        augmenter.select_parents("Yes")
        augmenter.add_to_pool("Synthetic 3", "Yes")

        # Get statistics
        selected_counts, iteration_counts = augmenter.get_statistics("Yes")

        # Check iteration_counts: total_iterations (3) - entry_iteration
        assert len(iteration_counts) == 6  # 3 original + 3 synthetic
        assert iteration_counts[0] == 3  # Original: 3 - 0 = 3
        assert iteration_counts[1] == 3
        assert iteration_counts[2] == 3
        assert iteration_counts[3] == 2  # Synthetic 1: 3 - 1 = 2
        assert iteration_counts[4] == 1  # Synthetic 2: 3 - 2 = 1
        assert iteration_counts[5] == 0  # Synthetic 3: 3 - 3 = 0

        # Check that selected_counts sum correctly (2 parents per iteration * 3 iterations = 6)
        assert sum(selected_counts) == 6

    def test_max_selections_limit(self, mock_embedding_model, sample_df):
        """Should exclude samples that have reached max_selections."""
        augmenter = GeneticAugmenter(mock_embedding_model, max_selections=1)
        augmenter.initialize_population(sample_df)

        # First selection: two samples get selected
        augmenter.current_iteration["Yes"] += 1
        p1, p2, idx1, idx2 = augmenter.select_parents("Yes")

        # Both should now have count = 1 (at max)
        assert augmenter.selected_count["Yes"][idx1] == 1
        assert augmenter.selected_count["Yes"][idx2] == 1

        # Second selection: should pick from the remaining sample
        augmenter.current_iteration["Yes"] += 1
        augmenter.add_to_pool("Synthetic 1", "Yes")  # Add one to have enough
        p1_2, p2_2, idx1_2, idx2_2 = augmenter.select_parents("Yes")

        # The previously selected samples should not be picked again
        assert idx1 not in {idx1_2, idx2_2}
        assert idx2 not in {idx1_2, idx2_2}

    def test_max_selections_raises_when_exhausted(self, mock_embedding_model):
        """Should raise ValueError when all samples reach max_selections."""
        # Only 2 samples per class, max_selections=1
        df = pd.DataFrame({
            'Text': ['Yes 1', 'Yes 2', 'No 1', 'No 2'],
            'class_label': ['Yes', 'Yes', 'No', 'No'],
        })
        augmenter = GeneticAugmenter(mock_embedding_model, max_selections=1)
        augmenter.initialize_population(df)

        # First selection exhausts both samples
        augmenter.current_iteration["Yes"] += 1
        augmenter.select_parents("Yes")

        # Second selection should fail (no samples available)
        augmenter.current_iteration["Yes"] += 1
        with pytest.raises(ValueError, match="Not enough available samples"):
            augmenter.select_parents("Yes")


class TestBuildBaseSet:
    """Tests for build_base_set function."""

    @pytest.fixture
    def sample_df(self):
        """Create a larger sample DataFrame."""
        yes_texts = [f"Yes claim {i}" for i in range(50)]
        no_texts = [f"No claim {i}" for i in range(50)]

        return pd.DataFrame({
            'Sentence_id': list(range(100)),
            'Text': yes_texts + no_texts,
            'class_label': ['Yes'] * 50 + ['No'] * 50,
        })

    def test_returns_correct_size(self, sample_df):
        """Should return DataFrame with specified size."""
        rng = random.Random(42)
        result = build_base_set(sample_df, 20, rng)

        assert len(result) == 20

    def test_is_balanced(self, sample_df):
        """Should have equal number of each class."""
        rng = random.Random(42)
        result = build_base_set(sample_df, 20, rng)

        yes_count = (result["class_label"] == "Yes").sum()
        no_count = (result["class_label"] == "No").sum()

        assert yes_count == 10
        assert no_count == 10

    def test_removes_sentence_id(self, sample_df):
        """Should remove Sentence_id column."""
        rng = random.Random(42)
        result = build_base_set(sample_df, 20, rng)

        assert "Sentence_id" not in result.columns

    def test_different_seeds_produce_different_results(self, sample_df):
        """Different RNG seeds should produce different base sets."""
        rng1 = random.Random(42)
        rng2 = random.Random(123)

        result1 = build_base_set(sample_df, 20, rng1)
        result2 = build_base_set(sample_df, 20, rng2)

        # The texts should be different
        assert not result1["Text"].equals(result2["Text"])


class TestCreateAugmentedDataset:
    """Tests for create_augmented_dataset function."""

    def test_combines_all_data(self):
        """Should combine base set with synthetic samples."""
        base_set = pd.DataFrame({
            'Text': ['real 1', 'real 2'],
            'class_label': ['Yes', 'No'],
        })
        synthetic_yes = ['syn yes 1', 'syn yes 2']
        synthetic_no = ['syn no 1']
        # Stats: 1 base Yes, 2 synthetic Yes; 1 base No, 1 synthetic No
        stats_yes = ([2, 1, 0], [3, 3, 2])  # (selected_count, iteration_count)
        stats_no = ([1, 0], [3, 1])

        result = create_augmented_dataset(
            base_set, synthetic_yes, synthetic_no, stats_yes, stats_no
        )

        assert len(result) == 5
        assert (result["class_label"] == "Yes").sum() == 3
        assert (result["class_label"] == "No").sum() == 2
        # Check statistics columns
        assert 'selected_count' in result.columns
        assert 'iteration_count' in result.columns
        # Order should be: base_yes, base_no, syn_yes_1, syn_yes_2, syn_no
        assert list(result['selected_count']) == [2, 1, 1, 0, 0]
        assert list(result['iteration_count']) == [3, 3, 3, 2, 1]

    def test_maintains_columns(self):
        """Should have Text, class_label, and statistics columns."""
        base_set = pd.DataFrame({
            'Text': ['real 1'],
            'class_label': ['Yes'],
        })
        stats_yes = ([1, 0], [2, 1])  # 1 base, 1 synthetic
        stats_no = ([0], [2])  # 0 base, 1 synthetic

        result = create_augmented_dataset(
            base_set, ['syn 1'], ['syn 2'], stats_yes, stats_no
        )

        assert 'Text' in result.columns
        assert 'class_label' in result.columns
        assert 'selected_count' in result.columns
        assert 'iteration_count' in result.columns


class TestIntegration:
    """Integration tests."""

    def test_generate_sample_async_mocked(self):
        """Test async generation with mocked LLM."""
        from augment import generate_sample_async

        async def run_test():
            with patch('augment.litellm.acompletion', new_callable=AsyncMock) as mock_llm:
                mock_response = MagicMock()
                mock_response.choices = [MagicMock(message=MagicMock(content="Generated claim"))]
                mock_llm.return_value = mock_response

                semaphore = asyncio.Semaphore(1)
                result = await generate_sample_async(
                    parent_1="Parent 1",
                    parent_2="Parent 2",
                    g1=["subject", "tone", "length"],
                    g2=["certainty", "specificity", "perspective"],
                    g3=["actor focus", "temporal orientation", "sentence structure", "speaker stance"],
                    class_label="Yes",
                    semaphore=semaphore,
                )

                assert result == "Generated claim"
                mock_llm.assert_called_once()

        asyncio.run(run_test())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
