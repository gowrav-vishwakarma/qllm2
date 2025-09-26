#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production System Test Script
Tests all production components to ensure everything works correctly
"""

import torch
import time
import sys
import os
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from production_trainer import ProductionTrainer, ProductionTokenizer, ProductionDataset
        print("✅ Production trainer imports successful")
    except ImportError as e:
        print(f"❌ Production trainer import failed: {e}")
        return False
    
    try:
        from brain_inspired_llm import BrainInspiredLLM, create_brain_inspired_model
        print("✅ Brain-inspired LLM imports successful")
    except ImportError as e:
        print(f"❌ Brain-inspired LLM import failed: {e}")
        return False
    
    try:
        from brain_inspired_trainer import BrainInspiredTrainingSystem
        print("✅ Brain-inspired trainer imports successful")
    except ImportError as e:
        print(f"❌ Brain-inspired trainer import failed: {e}")
        return False
    
    try:
        from evaluate_production import ProductionEvaluator
        print("✅ Production evaluator imports successful")
    except ImportError as e:
        print(f"❌ Production evaluator import failed: {e}")
        return False
    
    return True

def test_tokenizer():
    """Test the production tokenizer"""
    print("\n🧪 Testing production tokenizer...")
    
    try:
        from production_trainer import ProductionTokenizer
        
        # Create tokenizer
        tokenizer = ProductionTokenizer(vocab_size=256)
        print(f"✅ Tokenizer created with vocab size: {len(tokenizer)}")
        
        # Test encoding/decoding
        test_text = "The brain-inspired language model uses consciousness mechanisms."
        tokens = tokenizer.encode(test_text, max_length=50)
        decoded = tokenizer.decode(tokens)
        
        print(f"✅ Encoding/decoding test passed")
        print(f"   Original: {test_text}")
        print(f"   Decoded:  {decoded}")
        
        return True
        
    except Exception as e:
        print(f"❌ Tokenizer test failed: {e}")
        return False

def test_model_creation():
    """Test model creation and basic functionality"""
    print("\n🧪 Testing model creation...")
    
    try:
        from brain_inspired_llm import create_brain_inspired_model
        
        # Create small model for testing
        model = create_brain_inspired_model(vocab_size=256, dim=128, num_layers=2)
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test forward pass
        batch_size, seq_len = 2, 16
        x = torch.randint(0, 256, (batch_size, seq_len))
        
        with torch.no_grad():
            logits = model(x, training_step=100)
        
        print(f"✅ Forward pass successful, output shape: {logits.shape}")
        
        # Test consciousness state
        consciousness_state = model.get_consciousness_state(x)
        print(f"✅ Consciousness state retrieved: {consciousness_state['consciousness_weights'].shape}")
        
        # Test memory stats
        memory_stats = model.get_memory_stats()
        print(f"✅ Memory stats retrieved: {memory_stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        return False

def test_dataset():
    """Test production dataset"""
    print("\n🧪 Testing production dataset...")
    
    try:
        from production_trainer import ProductionTokenizer, ProductionDataset
        
        # Create tokenizer and dataset
        tokenizer = ProductionTokenizer(vocab_size=256)
        sample_texts = [
            "The brain-inspired language model uses consciousness mechanisms.",
            "Memory consolidation and retrieval are key components.",
            "Spiking neurons provide event-driven processing."
        ]
        
        dataset = ProductionDataset(
            texts=sample_texts,
            tokenizer=tokenizer,
            max_length=32,
            cache_size=100
        )
        
        print(f"✅ Dataset created with {len(dataset)} samples")
        
        # Test data loading
        sample = dataset[0]
        print(f"✅ Sample loaded: {sample['input_tokens'].shape}, {sample['target_tokens'].shape}")
        
        # Test cache stats
        cache_stats = dataset.get_cache_stats()
        print(f"✅ Cache stats: {cache_stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False

def test_training_system():
    """Test the production training system"""
    print("\n🧪 Testing production training system...")
    
    try:
        from production_trainer import ProductionTrainer
        
        # Create small configuration
        model_config = {
            'vocab_size': 256,
            'dim': 128,
            'num_layers': 2
        }
        
        training_config = {
            'batch_size': 2,
            'num_epochs': 1,
            'learning_rate': 1e-3,
            'max_length': 32,
            'weight_decay': 0.01,
            'grad_clip': 1.0,
            'num_workers': 0,
            'cache_size': 100
        }
        
        # Create trainer
        trainer = ProductionTrainer(model_config, training_config, "test_output")
        print(f"✅ Trainer created successfully")
        
        # Test data loader creation
        dataset_configs = [{'name': 'wikitext2', 'max_samples': 10}]
        train_loader, val_loader = trainer.create_data_loaders(dataset_configs)
        print(f"✅ Data loaders created: {len(train_loader)} train, {len(val_loader)} val batches")
        
        return True
        
    except Exception as e:
        print(f"❌ Training system test failed: {e}")
        return False

def test_generation():
    """Test text generation"""
    print("\n🧪 Testing text generation...")
    
    try:
        from brain_inspired_llm import create_brain_inspired_model
        from production_trainer import ProductionTokenizer
        
        # Create model and tokenizer
        model = create_brain_inspired_model(vocab_size=256, dim=128, num_layers=2)
        tokenizer = ProductionTokenizer(vocab_size=256)
        
        # Test generation
        prompt = "The brain-inspired"
        generated = trainer.generate_text(prompt, max_length=20, temperature=0.7)
        
        print(f"✅ Text generation successful")
        print(f"   Prompt: {prompt}")
        print(f"   Generated: {generated}")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        return False

def test_evaluation():
    """Test evaluation system"""
    print("\n🧪 Testing evaluation system...")
    
    try:
        from brain_inspired_llm import create_brain_inspired_model
        from production_trainer import ProductionTokenizer
        import tempfile
        
        # Create a dummy checkpoint for testing
        model = create_brain_inspired_model(vocab_size=256, dim=128, num_layers=2)
        tokenizer = ProductionTokenizer(vocab_size=256)
        
        # Create dummy checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_config': {'vocab_size': 256, 'dim': 128, 'num_layers': 2},
            'training_config': {'batch_size': 2, 'num_epochs': 1},
            'epoch': 1,
            'step': 100,
            'best_val_loss': 5.0
        }
        
        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(checkpoint, f.name)
            checkpoint_path = f.name
        
        # Test evaluator
        from evaluate_production import ProductionEvaluator
        evaluator = ProductionEvaluator(checkpoint_path, device='cpu')
        print(f"✅ Evaluator created successfully")
        
        # Clean up
        os.unlink(checkpoint_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        return False

def test_performance():
    """Test basic performance metrics"""
    print("\n🧪 Testing performance...")
    
    try:
        from brain_inspired_llm import create_brain_inspired_model
        
        # Create model
        model = create_brain_inspired_model(vocab_size=256, dim=128, num_layers=2)
        
        # Test forward pass speed
        batch_size, seq_len = 4, 32
        x = torch.randint(0, 256, (batch_size, seq_len))
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(x)
        
        # Time forward pass
        start_time = time.time()
        num_iterations = 100
        
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = model(x)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_iterations
        
        print(f"✅ Performance test completed")
        print(f"   Average forward pass time: {avg_time*1000:.2f}ms")
        print(f"   Throughput: {batch_size/avg_time:.1f} samples/second")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧠 BRAIN-INSPIRED LLM PRODUCTION SYSTEM TEST")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Tokenizer", test_tokenizer),
        ("Model Creation", test_model_creation),
        ("Dataset", test_dataset),
        ("Training System", test_training_system),
        ("Generation", test_generation),
        ("Evaluation", test_evaluation),
        ("Performance", test_performance)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"❌ {test_name} test FAILED with exception: {e}")
    
    print(f"\n{'='*60}")
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Production system is ready.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
