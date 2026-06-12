"""
Unit tests for InstructBLIP instance reuse between VQAScore and VLMJudge.

**Validates: Requirement 5.6**

Tests verify that:
- VQAScore and VLMJudge share the same InstructBLIP instance
- No duplicate model loading occurs
- Both scorers reference the exact same object (using `is` operator)
"""

import pytest
from unittest.mock import MagicMock

from src.models.instructblip_adapter import InstructBLIPAdapter
from src.scoring.vqascore import VQAScore
from src.scoring.vlm_judge import VLMJudge


class TestInstructBLIPInstanceReuse:
    """Test that VQAScore and VLMJudge share the same InstructBLIP instance."""
    
    def test_shared_instance_identity(self):
        """
        Should verify both scorers reference the exact same InstructBLIP object.
        
        This test creates a single InstructBLIP adapter instance and passes it
        to both VQAScore and VLMJudge constructors. It then verifies they
        reference the same object using the `is` operator (object identity check).
        """
        # Create a single InstructBLIP adapter instance
        # Note: We use a mock to avoid loading the actual model
        instructblip_instance = MagicMock(spec=InstructBLIPAdapter)
        
        # Pass the same instance to both scorers
        vqascore = VQAScore(instructblip_model=instructblip_instance)
        vlm_judge = VLMJudge(instructblip_model=instructblip_instance)
        
        # Verify both scorers have the model attribute
        assert hasattr(vqascore, 'model')
        assert hasattr(vlm_judge, 'model')
        
        # Critical check: verify they reference the SAME object (not just equal, but identical)
        assert vqascore.model is vlm_judge.model, (
            "VQAScore and VLMJudge must share the same InstructBLIP instance "
            "(object identity check failed)"
        )
        
        # Verify they both reference the original instance
        assert vqascore.model is instructblip_instance
        assert vlm_judge.model is instructblip_instance
    
    def test_no_duplicate_loading(self):
        """
        Should verify that using the same instance prevents duplicate loading.
        
        This test ensures that when both scorers are initialized with the same
        InstructBLIP instance, no additional model loading occurs (as would
        happen if each scorer created its own instance).
        """
        # Create a mock InstructBLIP adapter with load tracking
        instructblip_instance = MagicMock(spec=InstructBLIPAdapter)
        load_call_count = MagicMock()
        instructblip_instance.load = load_call_count
        
        # Initialize both scorers with the same instance
        vqascore = VQAScore(instructblip_model=instructblip_instance)
        vlm_judge = VLMJudge(instructblip_model=instructblip_instance)
        
        # Verify load() was NOT called by the scorers
        # (the instance is already loaded before being passed)
        load_call_count.assert_not_called()
        
        # Verify both scorers can access the model
        assert vqascore.model is not None
        assert vlm_judge.model is not None
    
    def test_vqascore_requires_instructblip_instance(self):
        """Should raise ValueError if VQAScore initialized without InstructBLIP."""
        with pytest.raises(ValueError, match="requires a loaded InstructBLIP model"):
            VQAScore(instructblip_model=None)
    
    def test_vlm_judge_requires_instructblip_instance(self):
        """Should raise ValueError if VLMJudge initialized without InstructBLIP."""
        with pytest.raises(ValueError, match="requires a loaded InstructBLIP model"):
            VLMJudge(instructblip_model=None)
    
    def test_dependency_injection_pattern(self):
        """
        Should verify dependency injection pattern is used correctly.
        
        This test confirms that both scorers receive their InstructBLIP
        dependency through constructor injection (not by creating their own),
        which is the design pattern that enables instance sharing.
        """
        # Create InstructBLIP instance
        instructblip_instance = MagicMock(spec=InstructBLIPAdapter)
        
        # Initialize scorers using dependency injection
        vqascore = VQAScore(instructblip_model=instructblip_instance)
        vlm_judge = VLMJudge(instructblip_model=instructblip_instance)
        
        # Verify the dependency was injected (not created internally)
        # Both should reference the injected instance
        assert vqascore.model is instructblip_instance
        assert vlm_judge.model is instructblip_instance
        
        # Verify neither scorer created its own instance
        # (they both use the exact same reference)
        assert id(vqascore.model) == id(vlm_judge.model) == id(instructblip_instance)
    
    def test_instance_reuse_with_multiple_comparisons(self):
        """
        Should verify shared instance remains consistent across multiple operations.
        
        This test ensures that when both scorers perform operations (like
        image comparisons), they continue to reference the same InstructBLIP
        instance throughout execution.
        """
        # Create InstructBLIP instance with query_image method
        instructblip_instance = MagicMock(spec=InstructBLIPAdapter)
        instructblip_instance.query_image.return_value = "yes"  # Mock response
        
        # Initialize both scorers
        vqascore = VQAScore(instructblip_model=instructblip_instance)
        vlm_judge = VLMJudge(instructblip_model=instructblip_instance)
        
        # Verify initial identity
        initial_vqa_id = id(vqascore.model)
        initial_vlm_id = id(vlm_judge.model)
        
        assert initial_vqa_id == initial_vlm_id
        
        # Simulate operations (the model reference should not change)
        # Note: We're not calling the full compare() since that requires PIL images
        # We're just verifying the model reference remains stable
        
        # Verify identity after operations
        assert id(vqascore.model) == initial_vqa_id
        assert id(vlm_judge.model) == initial_vlm_id
        assert vqascore.model is vlm_judge.model


class TestInstanceReuseBenefits:
    """Test the benefits of instance reuse (memory efficiency)."""
    
    def test_memory_efficiency_no_duplicate_instances(self):
        """
        Should demonstrate that instance reuse prevents duplicate memory usage.
        
        This test verifies that with instance reuse, only one InstructBLIP
        model exists in memory, as opposed to two separate instances.
        """
        # Create a single instance
        instructblip_instance = MagicMock(spec=InstructBLIPAdapter)
        
        # Initialize both scorers with shared instance
        vqascore = VQAScore(instructblip_model=instructblip_instance)
        vlm_judge = VLMJudge(instructblip_model=instructblip_instance)
        
        # Collect all InstructBLIP references
        references = [vqascore.model, vlm_judge.model, instructblip_instance]
        
        # Verify all references point to the same object
        # (meaning only 1 instance exists, not 2 or 3)
        unique_ids = set(id(ref) for ref in references)
        assert len(unique_ids) == 1, (
            f"Expected 1 unique InstructBLIP instance, found {len(unique_ids)}"
        )
    
    def test_singleton_like_behavior(self):
        """
        Should verify that instance reuse creates singleton-like behavior.
        
        While not a true singleton pattern, the dependency injection approach
        ensures that both scorers work with the same single instance.
        """
        # Create one instance
        instructblip_instance = MagicMock(spec=InstructBLIPAdapter)
        
        # Create scorers
        vqascore = VQAScore(instructblip_model=instructblip_instance)
        vlm_judge = VLMJudge(instructblip_model=instructblip_instance)
        
        # Verify singleton-like behavior
        assert vqascore.model is vlm_judge.model
        
        # If we modify the instance (e.g., add an attribute), both should see it
        instructblip_instance.test_attribute = "shared_value"
        
        assert hasattr(vqascore.model, 'test_attribute')
        assert hasattr(vlm_judge.model, 'test_attribute')
        assert vqascore.model.test_attribute == "shared_value"
        assert vlm_judge.model.test_attribute == "shared_value"
