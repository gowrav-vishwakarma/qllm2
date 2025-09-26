#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concept Layer Implementation - Phase 2.1
Provides semantic understanding and multilingual support through concept space mapping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConceptLayer(nn.Module):
    """
    Concept Layer Implementation - Phase 2.1
    Provides semantic understanding and multilingual support through concept space mapping
    """
    def __init__(self, vocab_size, concept_dim=256, num_concepts=512, num_layers=2, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.concept_dim = concept_dim
        self.num_concepts = num_concepts
        self.num_layers = num_layers
        
        # Word-to-concept mapping
        self.word_concept_map = nn.Embedding(vocab_size, num_concepts)
        nn.init.normal_(self.word_concept_map.weight, mean=0.0, std=0.02)
        
        # Concept embeddings (learnable concept representations)
        self.concept_embeddings = nn.Parameter(torch.randn(num_concepts, concept_dim) * 0.02)
        
        # Concept space transformations
        self.concept_projections = nn.ModuleList([
            nn.Linear(concept_dim, concept_dim) for _ in range(num_layers)
        ])
        
        # Concept attention mechanism
        self.concept_attention = nn.MultiheadAttention(
            embed_dim=concept_dim, 
            num_heads=8, 
            dropout=dropout,
            batch_first=True
        )
        
        # Concept-to-token mapping
        self.concept_to_token = nn.Linear(concept_dim, vocab_size, bias=False)
        
        # Normalization and dropout
        self.norm1 = nn.LayerNorm(concept_dim)
        self.norm2 = nn.LayerNorm(concept_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Semantic coherence layer
        self.semantic_coherence = nn.Linear(concept_dim, concept_dim)
        
        # Language-specific concept mappings (for multilingual support)
        self.language_gates = nn.Parameter(torch.randn(10, concept_dim) * 0.02)  # Support 10 languages
        
    def forward(self, token_embeddings, token_ids=None, language_id=0):
        """
        Forward pass through concept layer
        Args:
            token_embeddings: [batch_size, seq_len, embed_dim]
            token_ids: [batch_size, seq_len] - for concept mapping
            language_id: int - for language-specific processing
        """
        batch_size, seq_len, embed_dim = token_embeddings.shape
        
        # 1. Word-to-concept mapping
        if token_ids is not None:
            # Get concept weights for each token
            concept_weights = self.word_concept_map(token_ids)  # [batch_size, seq_len, num_concepts]
            
            # Apply concept embeddings
            concept_repr = torch.matmul(concept_weights, self.concept_embeddings)  # [batch_size, seq_len, concept_dim]
        else:
            # Fallback: project token embeddings to concept space
            concept_repr = torch.matmul(token_embeddings, self.concept_embeddings[:embed_dim, :].t())
        
        # 2. Language-specific concept processing
        if language_id < self.language_gates.shape[0]:
            language_gate = self.language_gates[language_id].unsqueeze(0).unsqueeze(0)  # [1, 1, concept_dim]
            concept_repr = concept_repr * torch.sigmoid(language_gate)
        
        # 3. Concept space transformations
        for i, projection in enumerate(self.concept_projections):
            residual = concept_repr
            concept_repr = self.norm1(concept_repr)
            concept_repr = projection(concept_repr)
            concept_repr = F.gelu(concept_repr)
            concept_repr = self.dropout(concept_repr)
            concept_repr = residual + concept_repr * 0.1  # Light residual connection
        
        # 4. Concept attention (self-attention in concept space)
        concept_repr_norm = self.norm2(concept_repr)
        attended_concepts, _ = self.concept_attention(
            concept_repr_norm, concept_repr_norm, concept_repr_norm
        )
        concept_repr = concept_repr + self.dropout(attended_concepts)
        
        # 5. Semantic coherence enhancement
        semantic_enhanced = self.semantic_coherence(concept_repr)
        concept_repr = concept_repr + torch.tanh(semantic_enhanced) * 0.1
        
        # 6. Concept-to-token mapping (for semantic loss calculation)
        concept_logits = self.concept_to_token(concept_repr)  # [batch_size, seq_len, vocab_size]
        
        return concept_repr, concept_logits
    
    def get_concept_representation(self, token_ids):
        """Get pure concept representation for analysis"""
        concept_weights = self.word_concept_map(token_ids)
        concept_repr = torch.matmul(concept_weights, self.concept_embeddings)
        return concept_repr
    
    def get_concept_attention_weights(self, token_ids):
        """Get attention weights for concept analysis"""
        concept_repr = self.get_concept_representation(token_ids)
        concept_repr_norm = self.norm2(concept_repr)
        _, attention_weights = self.concept_attention(
            concept_repr_norm, concept_repr_norm, concept_repr_norm
        )
        return attention_weights

class ConceptLoss(nn.Module):
    """
    Concept-based loss functions for semantic understanding
    """
    def __init__(self, concept_dim=256, num_concepts=512):
        super().__init__()
        self.concept_dim = concept_dim
        self.num_concepts = num_concepts
        
        # Concept coherence loss
        self.concept_coherence = nn.Linear(concept_dim, concept_dim)
        
        # Semantic similarity loss
        self.semantic_similarity = nn.CosineEmbeddingLoss()
        
    def forward(self, concept_repr, targets=None):
        """
        Calculate concept-based losses
        Args:
            concept_repr: [batch_size, seq_len, concept_dim]
            targets: [batch_size, seq_len] - target tokens
        """
        batch_size, seq_len, concept_dim = concept_repr.shape
        
        # 1. Concept coherence loss (encourage coherent concept representations)
        if seq_len > 1:
            # Calculate coherence between adjacent concept representations
            concept_diff = concept_repr[:, 1:] - concept_repr[:, :-1]
            coherence_loss = torch.mean(torch.norm(concept_diff, dim=-1))
        else:
            coherence_loss = torch.tensor(0.0, device=concept_repr.device)
        
        # 2. Concept diversity loss (encourage diverse concept usage)
        # Calculate variance across concept dimensions
        concept_variance = torch.var(concept_repr, dim=(0, 1))  # [concept_dim]
        diversity_loss = -torch.mean(concept_variance)  # Negative because we want to maximize variance
        
        # 3. Concept stability loss (encourage stable concept representations)
        if seq_len > 1:
            # Calculate stability across sequence
            concept_stability = torch.var(concept_repr, dim=1)  # [batch_size, concept_dim]
            stability_loss = torch.mean(concept_stability)
        else:
            stability_loss = torch.tensor(0.0, device=concept_repr.device)
        
        # Combined concept loss
        total_loss = (
            0.4 * coherence_loss +
            0.3 * diversity_loss +
            0.3 * stability_loss
        )
        
        return {
            'total': total_loss,
            'coherence': coherence_loss,
            'diversity': diversity_loss,
            'stability': stability_loss
        }

class MultilingualConceptProcessor(nn.Module):
    """
    Multilingual concept processing for cross-lingual understanding
    """
    def __init__(self, concept_dim=256, num_languages=10):
        super().__init__()
        self.concept_dim = concept_dim
        self.num_languages = num_languages
        
        # Language-specific concept mappings
        self.language_concept_maps = nn.ModuleList([
            nn.Linear(concept_dim, concept_dim) for _ in range(num_languages)
        ])
        
        # Universal concept space
        self.universal_concepts = nn.Parameter(torch.randn(concept_dim, concept_dim) * 0.02)
        
        # Cross-lingual attention
        self.cross_lingual_attention = nn.MultiheadAttention(
            embed_dim=concept_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
    def forward(self, concept_repr, language_id=0):
        """
        Process concepts for multilingual understanding
        Args:
            concept_repr: [batch_size, seq_len, concept_dim]
            language_id: int - target language ID
        """
        batch_size, seq_len, concept_dim = concept_repr.shape
        
        # 1. Language-specific concept transformation
        if language_id < len(self.language_concept_maps):
            language_specific = self.language_concept_maps[language_id](concept_repr)
        else:
            language_specific = concept_repr
        
        # 2. Universal concept mapping
        universal_concepts = torch.matmul(language_specific, self.universal_concepts)
        
        # 3. Cross-lingual attention
        attended_concepts, _ = self.cross_lingual_attention(
            universal_concepts, universal_concepts, universal_concepts
        )
        
        # 4. Combine language-specific and universal concepts
        combined_concepts = language_specific + attended_concepts * 0.5
        
        return combined_concepts
