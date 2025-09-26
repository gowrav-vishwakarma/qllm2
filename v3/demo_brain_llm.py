#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brain-Inspired LLM Demo
Demonstrates the trained v3 brain-inspired language model
"""

import torch
import json
import os
from brain_inspired_llm import create_brain_inspired_model
from brain_inspired_trainer import BrainInspiredTrainingSystem

def load_trained_model(checkpoint_path="checkpoints_brain_inspired/brain_inspired_model.pt"):
    """Load the trained brain-inspired model"""
    print("🧠 Loading trained brain-inspired model...")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        print("Please run training first: python train_brain_llm.py")
        return None
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_config = checkpoint['model_config']
    training_summary = checkpoint['training_summary']
    
    print(f"📊 Model config: {model_config}")
    print(f"📊 Training steps: {training_summary['training_steps']}")
    
    # Create model
    model = create_brain_inspired_model(
        vocab_size=model_config['vocab_size'],
        dim=model_config['dim'],
        num_layers=model_config['num_layers']
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✅ Model loaded successfully!")
    return model, model_config

def generate_text(model, prompt, max_length=100, temperature=0.7):
    """Generate text using the brain-inspired model"""
    print(f"🎯 Generating text with prompt: '{prompt}'")
    
    # Convert prompt to tokens
    tokens = [ord(c) for c in prompt]
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    
    generated = input_tensor.clone()
    
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            logits = model(generated)
            
            # Get next token probabilities
            next_token_logits = logits[0, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
    
    # Convert back to text
    generated_text = ''.join([chr(token.item()) for token in generated[0]])
    return generated_text[len(prompt):]

def analyze_consciousness(model, text):
    """Analyze consciousness state for given text"""
    print(f"🧠 Analyzing consciousness for: '{text[:50]}...'")
    
    # Convert to tokens
    tokens = [ord(c) for c in text[:64]]  # Limit to 64 chars
    if len(tokens) < 64:
        tokens.extend([0] * (64 - len(tokens)))
    else:
        tokens = tokens[:64]
    
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    
    with torch.no_grad():
        # Get consciousness state
        consciousness_state = model.get_consciousness_state(input_tensor)
        
        # Get memory stats
        memory_stats = model.get_memory_stats()
    
    print("📊 Consciousness Analysis:")
    print(f"   🎭 Consciousness weights shape: {consciousness_state['consciousness_weights'].shape}")
    print(f"   🌍 Global consciousness shape: {consciousness_state['global_consciousness'].shape}")
    print(f"   🧠 Memory retrieval: {consciousness_state['memory_retrieval']:.4f}")
    
    print("📊 Memory Statistics:")
    print(f"   💾 Short-term usage: {memory_stats['short_term_usage']:.4f}")
    print(f"   🧠 Long-term concepts: {memory_stats['long_term_concepts']:.4f}")
    print(f"   📚 Episodic memories: {memory_stats['episodic_memories']:.4f}")
    print(f"   ⚡ Learning efficiency: {memory_stats['learning_efficiency']:.4f}")
    print(f"   🔄 Adaptation rate: {memory_stats['adaptation_rate']:.4f}")
    
    return consciousness_state, memory_stats

def demo_brain_inspired_llm():
    """Main demo function"""
    print("🧠 BRAIN-INSPIRED LANGUAGE MODEL DEMO")
    print("=" * 50)
    
    # Load model
    model, model_config = load_trained_model()
    if model is None:
        return
    
    print(f"\n📊 Model Information:")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Vocabulary: {model_config['vocab_size']}")
    print(f"   Dimension: {model_config['dim']}")
    print(f"   Layers: {model_config['num_layers']}")
    
    # Demo prompts
    prompts = [
        "The brain-inspired",
        "Consciousness is",
        "Memory consolidation",
        "Artificial intelligence",
        "Human learning"
    ]
    
    print(f"\n🎯 Text Generation Demo:")
    print("-" * 30)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{i}. Prompt: '{prompt}'")
        generated = generate_text(model, prompt, max_length=50, temperature=0.7)
        print(f"   Generated: {generated[:100]}...")
        
        # Analyze consciousness for the prompt
        analyze_consciousness(model, prompt)
        print()
    
    print("🎉 Demo completed!")
    print("\n💡 Key Features Demonstrated:")
    print("   ✅ Brain-inspired text generation")
    print("   ✅ Consciousness state analysis")
    print("   ✅ Memory system statistics")
    print("   ✅ Human-like learning mechanisms")

if __name__ == "__main__":
    demo_brain_inspired_llm()
