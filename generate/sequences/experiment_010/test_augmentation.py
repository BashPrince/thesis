import pytest
import pandas as pd
from collections import Counter

from create_sequences import generate_synthetic_samples, parse_llm_response


def make_base_set(size: int) -> pd.DataFrame:
    """Create a mock base set with balanced classes."""
    half = size // 2
    data = {
        "Text": [f"text_{i}" for i in range(size)],
        "class_label": ["Yes"] * half + ["No"] * half,
    }
    return pd.DataFrame(data)


def identity_augmenter(texts: list[str], class_label: str) -> list[str]:
    """Augmenter that returns texts unchanged."""
    return texts


class TestTemplateDistribution:
    """Tests for even distribution of templates across synthetic samples."""

    def test_real_100_aug_800_uses_each_template_8_times(self):
        """With 100 templates and 800 synthetic samples, each template should be used 8 times."""
        base_set = make_base_set(100)
        synthetic_pool = {}
        samples = generate_synthetic_samples(
            base_set=base_set,
            aug_size=800,
            synthetic_pool=synthetic_pool,
            augmenter=identity_augmenter,
            batch_size=5,
        )

        assert len(samples) == 800

        # Count how many times each template is used
        template_counts = Counter(s["example"] for s in samples)

        # Each of the 100 templates should be used exactly 8 times
        assert len(template_counts) == 100, f"Expected 100 templates, got {len(template_counts)}"
        for template_idx, count in template_counts.items():
            assert count == 8, f"Template {template_idx} used {count} times, expected 8"

    def test_real_200_aug_800_uses_each_template_4_times(self):
        """
        When progressing from real_100 to real_200 with aug_800,
        each of the 200 templates should be used 4 times.

        This tests the reuse logic: after generating for (100, 800) and (100, 1600),
        when we get to (200, 800), the new templates (100-199) should also be used.
        """
        synthetic_pool = {}

        # Simulate the sequence: (100, 800), (100, 1600), (200, 800)
        # Step 1: real_100_aug_800
        base_100 = make_base_set(100)
        generate_synthetic_samples(
            base_set=base_100,
            aug_size=800,
            synthetic_pool=synthetic_pool,
            augmenter=identity_augmenter,
            batch_size=5,
        )

        # Step 2: real_100_aug_1600
        generate_synthetic_samples(
            base_set=base_100,
            aug_size=1600,
            synthetic_pool=synthetic_pool,
            augmenter=identity_augmenter,
            batch_size=5,
        )

        # Step 3: real_200_aug_800 - this is what we're testing
        base_200 = make_base_set(200)
        samples = generate_synthetic_samples(
            base_set=base_200,
            aug_size=800,
            synthetic_pool=synthetic_pool,
            augmenter=identity_augmenter,
            batch_size=5,
        )

        assert len(samples) == 800

        # Count how many times each template is used
        template_counts = Counter(s["example"] for s in samples)

        # Each of the 200 templates should be used exactly 4 times
        assert len(template_counts) == 200, (
            f"Expected 200 templates to be used, got {len(template_counts)}. "
            f"Templates used: {sorted(template_counts.keys())}"
        )
        for template_idx, count in template_counts.items():
            assert count == 4, f"Template {template_idx} used {count} times, expected 4"

    def test_real_200_aug_1600_uses_each_template_8_times(self):
        """
        For real_200_aug_1600, each of the 200 templates should be used 8 times.
        """
        synthetic_pool = {}

        # Simulate full sequence up to (200, 1600)
        base_100 = make_base_set(100)

        # (100, 800)
        generate_synthetic_samples(base_100, 800, synthetic_pool, identity_augmenter, 5)

        # (100, 1600)
        generate_synthetic_samples(base_100, 1600, synthetic_pool, identity_augmenter, 5)

        # (200, 800)
        base_200 = make_base_set(200)
        generate_synthetic_samples(base_200, 800, synthetic_pool, identity_augmenter, 5)

        # (200, 1600) - this is what we're testing
        samples = generate_synthetic_samples(
            base_set=base_200,
            aug_size=1600,
            synthetic_pool=synthetic_pool,
            augmenter=identity_augmenter,
            batch_size=5,
        )

        assert len(samples) == 1600

        template_counts = Counter(s["example"] for s in samples)

        assert len(template_counts) == 200, (
            f"Expected 200 templates, got {len(template_counts)}"
        )
        for template_idx, count in template_counts.items():
            assert count == 8, f"Template {template_idx} used {count} times, expected 8"

    def test_reuse_samples_are_included_in_larger_aug(self):
        """
        Samples from aug_800 should be a subset of samples from aug_1600
        for the same base size.
        """
        base_set = make_base_set(100)
        synthetic_pool = {}

        # Generate aug_800
        samples_800 = generate_synthetic_samples(
            base_set=base_set,
            aug_size=800,
            synthetic_pool=synthetic_pool,
            augmenter=identity_augmenter,
            batch_size=5,
        )

        # Generate aug_1600 (should include all of aug_800)
        samples_1600 = generate_synthetic_samples(
            base_set=base_set,
            aug_size=1600,
            synthetic_pool=synthetic_pool,
            augmenter=identity_augmenter,
            batch_size=5,
        )

        # All samples from aug_800 should appear in aug_1600
        texts_800 = {s["Text"] for s in samples_800}
        texts_1600 = {s["Text"] for s in samples_1600}

        assert texts_800.issubset(texts_1600), (
            "aug_800 samples should be a subset of aug_1600 samples"
        )


class TestClassRestrictedBatching:
    """Tests that batches contain only one class."""

    def test_batches_are_class_restricted(self):
        """Each batch passed to augmenter should contain only Yes or only No samples."""
        base_set = make_base_set(100)
        synthetic_pool = {}

        batches_seen = []

        def tracking_augmenter(texts: list[str], class_label: str) -> list[str]:
            batches_seen.append((texts, class_label))
            return texts

        generate_synthetic_samples(
            base_set=base_set,
            aug_size=100,
            synthetic_pool=synthetic_pool,
            augmenter=tracking_augmenter,
            batch_size=5,
        )

        # Check each batch
        for batch, class_label in batches_seen:
            # Extract indices from text names
            indices = [int(t.split("_")[1]) for t in batch]
            # Check if all from same class (0-49 = Yes, 50-99 = No)
            all_yes = all(idx < 50 for idx in indices)
            all_no = all(idx >= 50 for idx in indices)
            assert all_yes or all_no, (
                f"Batch contains mixed classes: {batch} (indices: {indices})"
            )
            # Also verify class_label matches
            if all_yes:
                assert class_label == "Yes", f"Expected Yes, got {class_label}"
            else:
                assert class_label == "No", f"Expected No, got {class_label}"


class TestParseLLMResponse:
    """Tests for LLM response parsing."""

    def test_parse_simple_list(self):
        """Parse a simple bullet list."""
        response = "- First item\n- Second item\n- Third item"
        result = parse_llm_response(response, 3)
        assert result == ["First item", "Second item", "Third item"]

    def test_parse_with_leading_whitespace(self):
        """Parse a list with leading whitespace on lines."""
        response = "  - First item\n  - Second item"
        result = parse_llm_response(response, 2)
        assert result == ["First item", "Second item"]

    def test_parse_with_extra_text(self):
        """Parse a response that has extra text before/after the list."""
        response = """Here are the augmented samples:

- Sample one with some content
- Sample two with more content
- Sample three is here

I hope these are helpful!"""
        result = parse_llm_response(response, 3)
        assert result == [
            "Sample one with some content",
            "Sample two with more content",
            "Sample three is here",
        ]

    def test_parse_strips_whitespace(self):
        """Parsed items should have whitespace stripped."""
        response = "-   Item with extra spaces   \n- Another item  "
        result = parse_llm_response(response, 2)
        assert result == ["Item with extra spaces", "Another item"]

    def test_parse_wrong_count_raises(self):
        """Raise ValueError if count doesn't match."""
        response = "- Item one\n- Item two"
        with pytest.raises(ValueError, match="Expected 3 samples, got 2"):
            parse_llm_response(response, 3)

    def test_parse_empty_response_raises(self):
        """Raise ValueError for empty response."""
        response = "No items here!"
        with pytest.raises(ValueError, match="Expected 2 samples, got 0"):
            parse_llm_response(response, 2)

    def test_parse_preserves_special_characters(self):
        """Special characters in items should be preserved."""
        response = "- Item with $100 and 50% off!\n- Another (item) [here]"
        result = parse_llm_response(response, 2)
        assert result == ["Item with $100 and 50% off!", "Another (item) [here]"]

    def test_parse_multiline_not_matched(self):
        """Lines without leading dash are not matched."""
        response = "- First item\nNot an item\n- Second item"
        result = parse_llm_response(response, 2)
        assert result == ["First item", "Second item"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
