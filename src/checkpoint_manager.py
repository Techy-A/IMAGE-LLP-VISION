"""
Checkpoint manager for tournament state persistence and recovery.

This module implements checkpoint saving and loading to enable tournament
recovery after interruptions or failures. Checkpoints include completed
matches, current ratings, pending pairs, and configuration snapshots.
"""

import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

from src.data_models import CheckpointState, MatchResult


logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manager for tournament checkpoint persistence.
    
    Handles saving and loading tournament state to enable recovery from
    interruptions. Checkpoints include:
    - Iteration number
    - Completed matches
    - Current Elo ratings
    - Pending image pairs
    - Configuration snapshot
    
    Attributes:
        checkpoint_dir: Directory for storing checkpoint files
        auto_backup: Whether to backup existing checkpoints before overwriting
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        auto_backup: bool = True
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory path for checkpoint files
            auto_backup: Whether to backup existing checkpoints (default: True)
        
        Raises:
            ValueError: If checkpoint_dir is invalid
        """
        if not checkpoint_dir:
            raise ValueError("checkpoint_dir cannot be empty")
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.auto_backup = auto_backup
        
        # Create checkpoint directory if it doesn't exist
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"CheckpointManager initialized with directory: {self.checkpoint_dir}"
        )
    
    def save_checkpoint(
        self,
        state: CheckpointState,
        iteration: int,
        filename: Optional[str] = None
    ) -> str:
        """
        Save tournament state to checkpoint file.
        
        Args:
            state: CheckpointState containing tournament state
            iteration: Current iteration number
            filename: Optional custom filename (auto-generated if None)
        
        Returns:
            Path to saved checkpoint file
        
        Raises:
            RuntimeError: If checkpoint save fails
        """
        try:
            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"checkpoint_iter{iteration:06d}_{timestamp}.json"
            
            checkpoint_path = self.checkpoint_dir / filename
            
            # Backup existing checkpoint if auto_backup enabled
            if self.auto_backup and checkpoint_path.exists():
                self._backup_checkpoint(checkpoint_path)
            
            # Serialize checkpoint state
            checkpoint_data = self._serialize_checkpoint(state)
            
            # Write to file with atomic write (write to temp, then rename)
            temp_path = checkpoint_path.with_suffix('.tmp')
            
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            temp_path.replace(checkpoint_path)
            
            logger.info(
                f"Checkpoint saved: {checkpoint_path} "
                f"(iteration {iteration}, "
                f"{len(state.completed_matches)} matches, "
                f"{len(state.pending_pairs)} pending)"
            )
            
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise RuntimeError(f"Checkpoint save failed: {e}") from e
    
    def load_checkpoint(self, checkpoint_path: str) -> CheckpointState:
        """
        Load tournament state from checkpoint file.
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            CheckpointState containing restored tournament state
        
        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            RuntimeError: If checkpoint loading or validation fails
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found: {checkpoint_path}"
            )
        
        try:
            # Read checkpoint file
            with open(checkpoint_path, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            # Deserialize checkpoint state
            state = self._deserialize_checkpoint(checkpoint_data)
            
            logger.info(
                f"Checkpoint loaded: {checkpoint_path} "
                f"(iteration {state.iteration}, "
                f"{len(state.completed_matches)} matches, "
                f"{len(state.pending_pairs)} pending)"
            )
            
            return state
            
        except json.JSONDecodeError as e:
            # Checkpoint file is corrupted
            logger.error(f"Checkpoint file corrupted: {e}")
            self._mark_corrupted(checkpoint_path)
            raise RuntimeError(
                f"Checkpoint file corrupted: {checkpoint_path}. "
                f"Marked as .corrupted for inspection."
            ) from e
        
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise RuntimeError(f"Checkpoint load failed: {e}") from e
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all checkpoint files in checkpoint directory.
        
        Returns:
            List of checkpoint info dictionaries with keys:
            - path: Full path to checkpoint file
            - filename: Checkpoint filename
            - size: File size in bytes
            - modified: Last modified timestamp
        """
        checkpoints = []
        
        for path in self.checkpoint_dir.glob("checkpoint_*.json"):
            try:
                stat = path.stat()
                checkpoints.append({
                    'path': str(path),
                    'filename': path.name,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime)
                })
            except Exception as e:
                logger.warning(f"Failed to stat checkpoint {path}: {e}")
        
        # Sort by modification time (newest first)
        checkpoints.sort(key=lambda x: x['modified'], reverse=True)
        
        return checkpoints
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Get path to most recent checkpoint file.
        
        Returns:
            Path to latest checkpoint, or None if no checkpoints exist
        """
        checkpoints = self.list_checkpoints()
        
        if not checkpoints:
            return None
        
        return checkpoints[0]['path']
    
    def delete_checkpoint(self, checkpoint_path: str) -> None:
        """
        Delete a checkpoint file.
        
        Args:
            checkpoint_path: Path to checkpoint file to delete
        
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            RuntimeError: If deletion fails
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found: {checkpoint_path}"
            )
        
        try:
            checkpoint_path.unlink()
            logger.info(f"Deleted checkpoint: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to delete checkpoint: {e}")
            raise RuntimeError(f"Checkpoint deletion failed: {e}") from e
    
    def _serialize_checkpoint(self, state: CheckpointState) -> Dict[str, Any]:
        """
        Serialize checkpoint state to JSON-compatible dict.
        
        Args:
            state: CheckpointState to serialize
        
        Returns:
            JSON-compatible dictionary
        """
        # Serialize completed matches
        matches_data = []
        for match in state.completed_matches:
            matches_data.append({
                'image_a_id': match.image_a_id,
                'image_b_id': match.image_b_id,
                'scores_by_scorer': match.scores_by_scorer,
                'ensemble_score': match.ensemble_score,
                'winner_id': match.winner_id,
                'timestamp': match.timestamp.isoformat()
            })
        
        return {
            'version': '1.0',
            'iteration': state.iteration,
            'completed_matches': matches_data,
            'current_elo_ratings': state.current_elo_ratings,
            'pending_pairs': state.pending_pairs,
            'timestamp': state.timestamp.isoformat(),
            'config_snapshot': state.config_snapshot
        }
    
    def _deserialize_checkpoint(self, data: Dict[str, Any]) -> CheckpointState:
        """
        Deserialize checkpoint data to CheckpointState.
        
        Args:
            data: JSON checkpoint data
        
        Returns:
            CheckpointState object
        
        Raises:
            ValueError: If checkpoint data is invalid
        """
        # Validate required fields
        required_fields = [
            'iteration', 'completed_matches', 'current_elo_ratings',
            'pending_pairs', 'timestamp', 'config_snapshot'
        ]
        
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field in checkpoint: {field}")
        
        # Deserialize completed matches
        completed_matches = []
        for match_data in data['completed_matches']:
            match = MatchResult(
                image_a_id=match_data['image_a_id'],
                image_b_id=match_data['image_b_id'],
                scores_by_scorer=match_data['scores_by_scorer'],
                ensemble_score=match_data['ensemble_score'],
                winner_id=match_data['winner_id'],
                timestamp=datetime.fromisoformat(match_data['timestamp'])
            )
            completed_matches.append(match)
        
        # Create CheckpointState
        state = CheckpointState(
            iteration=data['iteration'],
            completed_matches=completed_matches,
            current_elo_ratings=data['current_elo_ratings'],
            pending_pairs=[tuple(pair) for pair in data['pending_pairs']],
            timestamp=datetime.fromisoformat(data['timestamp']),
            config_snapshot=data['config_snapshot']
        )
        
        return state
    
    def _backup_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Create backup of existing checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint to backup
        """
        try:
            backup_path = checkpoint_path.with_suffix('.backup')
            checkpoint_path.replace(backup_path)
            logger.debug(f"Backed up checkpoint to {backup_path}")
            
        except Exception as e:
            logger.warning(f"Failed to backup checkpoint: {e}")
    
    def _mark_corrupted(self, checkpoint_path: Path) -> None:
        """
        Mark checkpoint as corrupted by renaming.
        
        Args:
            checkpoint_path: Path to corrupted checkpoint
        """
        try:
            corrupted_path = checkpoint_path.with_suffix('.corrupted')
            checkpoint_path.replace(corrupted_path)
            logger.info(f"Marked corrupted checkpoint: {corrupted_path}")
            
        except Exception as e:
            logger.warning(f"Failed to mark checkpoint as corrupted: {e}")
    
    def __repr__(self) -> str:
        """String representation of CheckpointManager."""
        return (
            f"CheckpointManager(checkpoint_dir='{self.checkpoint_dir}', "
            f"auto_backup={self.auto_backup})"
        )
