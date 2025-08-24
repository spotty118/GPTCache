#!/usr/bin/env python3
"""
Intelligent Context Memory Compression for GPTCache
Compresses conversation history while preserving key information
"""
import time
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import re

import numpy as np


@dataclass
class MessageImportance:
    """Importance score for a conversation message."""
    message_index: int
    timestamp: float
    content: str
    role: str
    importance_score: float
    semantic_embedding: Optional[np.ndarray] = None
    compression_tier: str = "full"  # full, summary, compressed, omit


@dataclass
class CompressionResult:
    """Result of context compression."""
    original_messages: List[Dict[str, Any]]
    compressed_messages: List[Dict[str, Any]]
    compression_summary: str
    tokens_saved: int
    compression_ratio: float
    preserved_information: List[str]


class ContextCompressionEngine:
    """
    Intelligent context compression engine that:
    1. Analyzes conversation importance using embeddings
    2. Preserves critical information
    3. Compresses less important context
    4. Maintains conversation flow
    """
    
    def __init__(self, 
                 max_context_tokens: int = 4000,
                 recent_messages_preserve: int = 5,
                 importance_threshold: float = 0.7,
                 embedding_model=None,
                 llm_client=None):
        """
        Initialize context compression engine.
        
        :param max_context_tokens: Maximum tokens to maintain in context
        :param recent_messages_preserve: Always preserve this many recent messages
        :param importance_threshold: Threshold for message importance (0-1)
        :param embedding_model: Embedding model for semantic analysis
        :param llm_client: LLM client for generating summaries
        """
        self.max_context_tokens = max_context_tokens
        self.recent_messages_preserve = recent_messages_preserve
        self.importance_threshold = importance_threshold
        self.embedding_model = embedding_model
        self.llm_client = llm_client
        
        # Importance factors
        self.recency_weight = 0.3
        self.semantic_weight = 0.4
        self.role_weight = 0.2
        self.length_weight = 0.1
        
        # Cache for embeddings and importance scores
        self.message_cache = {}
    
    def compress_context(self, messages: List[Dict[str, Any]], 
                        current_tokens: int,
                        target_max_tokens: Optional[int] = None,
                        always_compress: bool = False) -> CompressionResult:
        """
        Compress conversation context intelligently.

        :param messages: List of conversation messages
        :param current_tokens: Current token count (estimated)
        :param target_max_tokens: Dynamic per-request target budget to fit into (defaults to engine max)
        :param always_compress: If True, perform compression even when under budget (light-weight savings)
        :return: Compression result with optimized context
        """
        target = target_max_tokens if target_max_tokens is not None else self.max_context_tokens

        if not always_compress and current_tokens <= target:
            return CompressionResult(
                original_messages=messages,
                compressed_messages=messages,
                compression_summary="No compression needed",
                tokens_saved=0,
                compression_ratio=1.0,
                preserved_information=[]
            )
        
        # Analyze message importance
        importance_scores = self._analyze_message_importance(messages)
        
        # Determine initial compression strategy
        compression_plan = self._create_compression_plan(
            importance_scores, current_tokens
        )

        # Iteratively tighten compression until we meet the target budget
        compression_plan = self._tighten_until_budget(
            messages=messages,
            plan=compression_plan,
            importance_scores=importance_scores,
            target_max_tokens=target
        )
        
        # Apply compression
        compressed_messages = self._apply_compression(
            messages, compression_plan
        )
        
        # Calculate results
        original_token_count = self._estimate_tokens(messages)
        compressed_token_count = self._estimate_tokens(compressed_messages)
        tokens_saved = original_token_count - compressed_token_count
        compression_ratio = compressed_token_count / original_token_count if original_token_count > 0 else 1.0
        
        return CompressionResult(
            original_messages=messages,
            compressed_messages=compressed_messages,
            compression_summary=self._generate_compression_summary(compression_plan),
            tokens_saved=tokens_saved,
            compression_ratio=compression_ratio,
            preserved_information=self._extract_preserved_info(compression_plan)
        )
    
    def _analyze_message_importance(self, messages: List[Dict[str, Any]]) -> List[MessageImportance]:
        """Analyze the importance of each message."""
        importance_scores = []
        current_time = time.time()
        
        for i, message in enumerate(messages):
            content = message.get('content', '')
            if isinstance(content, list):
                # Handle cases where content is a list of segments (e.g., from vision models)
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                content = " ".join(text_parts)
            role = message.get('role', 'user')
            timestamp = message.get('timestamp', current_time - (len(messages) - i) * 60)
            
            # Calculate importance factors
            recency_score = self._calculate_recency_score(timestamp, current_time)
            semantic_score = self._calculate_semantic_importance(content, messages)
            role_score = self._calculate_role_importance(role)
            length_score = self._calculate_length_importance(content)
            
            # Weighted importance score
            importance = (
                self.recency_weight * recency_score +
                self.semantic_weight * semantic_score +
                self.role_weight * role_score +
                self.length_weight * length_score
            )
            
            # Get embedding for semantic analysis
            embedding = None
            if self.embedding_model and content.strip():
                try:
                    embedding = self.embedding_model.to_embeddings(content)
                except Exception:
                    embedding = None
            
            importance_scores.append(MessageImportance(
                message_index=i,
                timestamp=timestamp,
                content=content,
                role=role,
                importance_score=importance,
                semantic_embedding=embedding
            ))
        
        return importance_scores
    
    def _calculate_recency_score(self, timestamp: float, current_time: float) -> float:
        """Calculate recency score (newer = more important)."""
        age_hours = (current_time - timestamp) / 3600
        # Exponential decay with 12-hour half-life
        return max(0.1, min(1.0, np.exp(-age_hours / 12)))
    
    def _calculate_semantic_importance(self, content: str, all_messages: List[Dict]) -> float:
        """Calculate semantic importance based on content analysis."""
        # Length factor
        length_factor = min(1.0, len(content) / 200)
        
        # Question/instruction factor
        question_factor = 1.3 if isinstance(content, str) and any(word in content.lower() for word in
                                   ['?', 'how', 'what', 'why', 'when', 'where', 'please']) else 1.0
        
        # Important keywords factor
        important_keywords = [
            'important', 'critical', 'remember', 'note', 'key', 'main',
            'summary', 'conclusion', 'result', 'error', 'problem', 'solution'
        ]
        keyword_factor = 1.2 if isinstance(content, str) and any(keyword in content.lower() for keyword in important_keywords) else 1.0
        
        # Code/technical content factor
        code_factor = 1.2 if isinstance(content, str) and any(marker in content for marker in ['```', 'def ', 'class ', 'import ', '()']) else 1.0
        
        base_score = length_factor * question_factor * keyword_factor * code_factor
        return min(1.0, base_score)
    
    def _calculate_role_importance(self, role: str) -> float:
        """Calculate importance based on message role."""
        role_weights = {
            'system': 0.9,      # System messages are usually important
            'assistant': 0.8,   # AI responses contain valuable information
            'user': 0.7,        # User input is important but can be compressed
            'function': 0.6,    # Function calls can often be summarized
        }
        return role_weights.get(role, 0.5)
    
    def _calculate_length_importance(self, content: str) -> float:
        """Calculate importance based on content length."""
        length = len(content)
        if length < 20:
            return 0.3  # Very short messages are often less important
        elif length < 100:
            return 0.6  # Medium messages
        elif length < 500:
            return 0.8  # Longer messages often contain more information
        else:
            return 1.0  # Very long messages are usually important
    
    def _create_compression_plan(self, importance_scores: List[MessageImportance], 
                               current_tokens: int) -> Dict[int, str]:
        """Create a plan for how to compress each message."""
        plan = {}
        
        # Always preserve recent messages
        total_messages = len(importance_scores)
        preserve_start = max(0, total_messages - self.recent_messages_preserve)
        
        for i, importance in enumerate(importance_scores):
            if i >= preserve_start:
                # Preserve recent messages in full
                plan[i] = "full"
            elif importance.importance_score >= 0.7:
                # High importance - preserve in full
                plan[i] = "full"
            elif importance.importance_score >= 0.5:
                # Medium importance - create summary
                plan[i] = "summary"
            elif importance.importance_score >= 0.2:
                # Low importance - compress heavily
                plan[i] = "compressed"
            else:
                # Very low importance - omit but mention in summary
                plan[i] = "omit"
        
        return plan
    
    def _apply_compression(self, messages: List[Dict[str, Any]], 
                          plan: Dict[int, str]) -> List[Dict[str, Any]]:
        """Apply the compression plan to messages."""
        compressed_messages = []
        omitted_count = 0
        
        for i, message in enumerate(messages):
            compression_type = plan.get(i, "full")
            
            if compression_type == "full":
                compressed_messages.append(message)
            elif compression_type == "summary":
                summary = self._create_message_summary(message)
                compressed_messages.append({
                    "role": message.get("role", "user"),
                    "content": f"[Summary] {summary}",
                    "original_length": len(message.get("content", "")),
                    "compressed": True
                })
            elif compression_type == "compressed":
                compressed = self._compress_message(message)
                compressed_messages.append({
                    "role": message.get("role", "user"),
                    "content": f"[Compressed] {compressed}",
                    "original_length": len(message.get("content", "")),
                    "compressed": True
                })
            else:  # omit
                omitted_count += 1
        
        # Add summary of omitted messages if any
        if omitted_count > 0:
            insert_pos = max(0, len(compressed_messages) - self.recent_messages_preserve) if self.recent_messages_preserve > 0 else len(compressed_messages)
            insert_pos = min(insert_pos, len(compressed_messages))
            compressed_messages.insert(insert_pos, {
                "role": "system",
                "content": f"[Context Note] {omitted_count} less relevant messages omitted from conversation history.",
                "compressed": True
            })
        
        return compressed_messages
    
    def _create_message_summary(self, message: Dict[str, Any]) -> str:
        """Create a summary of a message."""
        content = message.get("content", "")
        role = message.get("role", "user")
        
        # Extract key information
        sentences = content.split('. ')
        
        if len(sentences) <= 2:
            return content[:25] + "..." if len(content) > 25 else content
        
        # Keep first and last sentences, or most important ones
        important_sentences = []
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in 
                  ['important', 'key', 'main', 'result', 'conclusion', '?']):
                important_sentences.append(sentence)
        
        if important_sentences:
            summary = '. '.join(important_sentences[:2])
        else:
            summary = f"{sentences[0]}. [...] {sentences[-1]}"
        
        return summary[:150] + "..." if len(summary) > 150 else summary
    
    def _compress_message(self, message: Dict[str, Any]) -> str:
        """Heavily compress a message."""
        content = message.get("content", "")
        
        # Extract key phrases and entities
        key_phrases = re.findall(r'\b[A-Z][a-z]+\b|\b\d+\b|[?!]', content)
        
        # Create very short summary
        if len(content) > 50:
            words = content.split()
            compressed = ' '.join(words[:5] + ['...'] + words[-3:])
        else:
            compressed = content
        
        return compressed[:15] + "..." if len(compressed) > 15 else compressed
    
    def _generate_compression_summary(self, plan: Dict[int, str]) -> str:
        """Generate a summary of the compression performed."""
        full_count = sum(1 for action in plan.values() if action == "full")
        summary_count = sum(1 for action in plan.values() if action == "summary")
        compressed_count = sum(1 for action in plan.values() if action == "compressed")
        omitted_count = sum(1 for action in plan.values() if action == "omit")
        
        return (f"Compression applied: {full_count} messages preserved, "
                f"{summary_count} summarized, {compressed_count} compressed, "
                f"{omitted_count} omitted")
    
    def _extract_preserved_info(self, plan: Dict[int, str]) -> List[str]:
        """Extract information about what was preserved."""
        info = []
        for action in plan.values():
            if action == "full":
                info.append("Full conversation context")
            elif action == "summary":
                info.append("Key information summaries")
        return info
    
    def _estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate token count for messages."""
        total_chars = sum(len(msg.get("content", "")) for msg in messages)
        # Rough estimation: ~4 characters per token
        return total_chars // 4

    def _estimate_tokens_per_message(self, messages: List[Dict[str, Any]]) -> List[int]:
        """Estimate tokens per message with a small overhead for role/formatting."""
        est = []
        for msg in messages:
            content_len = len(msg.get("content", "") or "")
            role_len = len(msg.get("role", "") or "")
            # ~4 chars per token plus overhead ~4 tokens
            est.append((content_len // 4) + (role_len // 4) + 4)
        return est

    def _tighten_until_budget(
        self,
        messages: List[Dict[str, Any]],
        plan: Dict[int, str],
        importance_scores: List[MessageImportance],
        target_max_tokens: int
    ) -> Dict[int, str]:
        """Iteratively tighten compression plan until compressed context fits the target budget."""
        if target_max_tokens <= 0:
            return plan

        # Helper ordering for degradation
        degrade_order = {"full": "summary", "summary": "compressed", "compressed": "omit", "omit": "omit"}

        # Prepare a lookup for importance by index
        importance_by_index = {imp.message_index: imp.importance_score for imp in importance_scores}

        # Always preserve last N recent messages in 'full'
        total_messages = len(importance_scores)
        preserve_start = max(0, total_messages - self.recent_messages_preserve)

        def can_degrade(idx: int) -> bool:
            if idx >= preserve_start:
                return False  # keep very recent context intact
            current = plan.get(idx, "full")
            return current in ("full", "summary", "compressed")

        # Start with current plan and keep degrading least important messages first
        while True:
            compressed_messages = self._apply_compression(messages, plan)
            compressed_tokens = self._estimate_tokens(compressed_messages)
            if compressed_tokens <= target_max_tokens:
                break  # budget satisfied

            # Choose next candidate to degrade (lowest importance first)
            candidates = [
                idx for idx in range(total_messages)
                if can_degrade(idx)
            ]
            if not candidates:
                # Cannot degrade further; break to avoid infinite loop
                break

            # Sort by importance ascending (less important first), then by age (older first)
            candidates.sort(key=lambda idx: (importance_by_index.get(idx, 0.0), idx))

            progressed = False
            for idx in candidates:
                current = plan.get(idx, "full")
                next_level = degrade_order[current]
                if next_level != current:
                    plan[idx] = next_level
                    progressed = True
                    break  # degrade one step, then re-evaluate tokens

            if not progressed:
                break  # nothing left to do

        return plan


class AdaptiveContextManager:
    """
    Adaptive context manager that learns from usage patterns
    and optimizes compression strategies over time.
    """
    
    def __init__(self, compression_engine: ContextCompressionEngine):
        self.compression_engine = compression_engine
        self.usage_history = []
        self.compression_effectiveness = {}
        
    def manage_context(self, messages: List[Dict[str, Any]], 
                      max_tokens: int,
                      always_compress: bool = False) -> CompressionResult:
        """
        Manage context with adaptive learning.
        
        :param messages: Conversation messages
        :param max_tokens: Maximum token limit (target budget to fit)
        :param always_compress: If True, apply compression even when under budget
        :return: Optimized compression result
        """
        current_tokens = self.compression_engine._estimate_tokens(messages)
        
        if not always_compress and current_tokens <= max_tokens:
            return CompressionResult(
                original_messages=messages,
                compressed_messages=messages,
                compression_summary="No compression needed",
                tokens_saved=0,
                compression_ratio=1.0,
                preserved_information=[]
            )
        
        # Apply compression with dynamic budgeting
        result = self.compression_engine.compress_context(
            messages=messages,
            current_tokens=current_tokens,
            target_max_tokens=max_tokens,
            always_compress=always_compress
        )
        
        # Track usage for learning
        self._track_compression_usage(result)
        
        return result
    
    def _track_compression_usage(self, result: CompressionResult):
        """Track compression usage for adaptive learning."""
        self.usage_history.append({
            "timestamp": time.time(),
            "compression_ratio": result.compression_ratio,
            "tokens_saved": result.tokens_saved,
            "effectiveness": self._calculate_effectiveness(result)
        })
        
        # Keep only recent history
        if len(self.usage_history) > 100:
            self.usage_history = self.usage_history[-100:]
    
    def _calculate_effectiveness(self, result: CompressionResult) -> float:
        """Calculate compression effectiveness score."""
        # Effectiveness based on compression ratio and information preservation
        ratio_score = 1.0 - result.compression_ratio  # Higher compression is better
        preservation_score = len(result.preserved_information) / 10.0  # More preservation is better
        
        return (ratio_score + preservation_score) / 2.0
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression performance statistics."""
        if not self.usage_history:
            return {"message": "No compression history available"}
        
        recent_history = self.usage_history[-20:]  # Last 20 compressions
        
        avg_ratio = sum(h["compression_ratio"] for h in recent_history) / len(recent_history)
        avg_tokens_saved = sum(h["tokens_saved"] for h in recent_history) / len(recent_history)
        avg_effectiveness = sum(h["effectiveness"] for h in recent_history) / len(recent_history)
        
        return {
            "total_compressions": len(self.usage_history),
            "avg_compression_ratio": round(avg_ratio, 3),
            "avg_tokens_saved": round(avg_tokens_saved, 1),
            "avg_effectiveness": round(avg_effectiveness, 3),
            "recent_compressions": len(recent_history)
        }


# Factory functions for easy integration
def create_context_compressor(embedding_model=None, 
                            llm_client=None,
                            max_tokens: int = 4000) -> ContextCompressionEngine:
    """Create a context compression engine."""
    return ContextCompressionEngine(
        max_context_tokens=max_tokens,
        embedding_model=embedding_model,
        llm_client=llm_client
    )


def create_adaptive_context_manager(embedding_model=None,
                                  llm_client=None,
                                  max_tokens: int = 4000) -> AdaptiveContextManager:
    """Create an adaptive context manager."""
    try:
        engine = create_context_compressor(embedding_model, llm_client, max_tokens)
        return AdaptiveContextManager(engine)
    except Exception as e:
        print(f"Warning: Failed to create compression engine with embedding model: {e}")
        print("Falling back to basic compression without semantic analysis.")
        # Create engine without embedding model to avoid dependency issues
        engine = create_context_compressor(embedding_model=None, llm_client=None, max_tokens=max_tokens)
        return AdaptiveContextManager(engine)