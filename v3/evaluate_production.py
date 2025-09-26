#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production Evaluation Script for Brain-Inspired LLM
Comprehensive evaluation with multiple metrics and comparisons
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("⚠️ Matplotlib/seaborn not available, visualizations disabled")
from tqdm import tqdm

# Import our components
from brain_inspired_llm import BrainInspiredLLM, create_brain_inspired_model
from production_trainer import ProductionTokenizer

class ProductionEvaluator:
    """Comprehensive evaluator for brain-inspired LLM"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        self.device = self._setup_device(device)
        self.model_path = Path(model_path)
        
        # Load model and config
        self.checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model_config = self.checkpoint['model_config']
        self.training_config = self.checkpoint['training_config']
        
        # Create model
        self.model = create_brain_inspired_model(
            vocab_size=self.model_config['vocab_size'],
            dim=self.model_config['dim'],
            num_layers=self.model_config['num_layers']
        )
        
        # Load weights
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Create tokenizer
        self.tokenizer = ProductionTokenizer(
            vocab_size=self.model_config['vocab_size']
        )
        
        print(f"✅ Model loaded from {self.model_path}")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"📊 Training epochs: {self.checkpoint.get('epoch', 'unknown')}")
        print(f"📊 Best validation loss: {self.checkpoint.get('best_val_loss', 'unknown')}")
    
    def _setup_device(self, device: str) -> str:
        """Setup device for evaluation"""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        return device
    
    def evaluate_perplexity(self, texts: List[str], max_length: int = 512) -> Dict[str, float]:
        """Evaluate perplexity on given texts"""
        print("📊 Evaluating perplexity...")
        
        total_loss = 0.0
        total_tokens = 0
        perplexities = []
        
        with torch.no_grad():
            for text in tqdm(texts, desc="Computing perplexity"):
                # Tokenize text
                tokens = self.tokenizer.encode(text, max_length=max_length)
                
                if len(tokens) < 2:
                    continue
                
                # Create input and target
                input_tokens = torch.tensor(tokens[:-1], dtype=torch.long).unsqueeze(0).to(self.device)
                target_tokens = torch.tensor(tokens[1:], dtype=torch.long).unsqueeze(0).to(self.device)
                
                # Forward pass
                logits = self.model(input_tokens)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_tokens.view(-1),
                    reduction='sum'
                )
                
                # Update totals
                total_loss += loss.item()
                total_tokens += len(tokens) - 1
                
                # Individual perplexity
                perplexity = torch.exp(loss / (len(tokens) - 1)).item()
                perplexities.append(perplexity)
        
        # Overall metrics
        avg_loss = total_loss / total_tokens
        overall_perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'overall_perplexity': overall_perplexity,
            'average_loss': avg_loss,
            'total_tokens': total_tokens,
            'perplexity_std': np.std(perplexities),
            'perplexity_median': np.median(perplexities),
            'perplexity_min': np.min(perplexities),
            'perplexity_max': np.max(perplexities)
        }
    
    def evaluate_generation_quality(self, prompts: List[str], max_length: int = 100, 
                                  temperature: float = 0.7) -> Dict[str, Any]:
        """Evaluate text generation quality"""
        print("🎯 Evaluating generation quality...")
        
        generated_texts = []
        generation_times = []
        
        for prompt in tqdm(prompts, desc="Generating text"):
            start_time = time.time()
            generated = self._generate_text(prompt, max_length, temperature)
            generation_time = time.time() - start_time
            
            generated_texts.append(generated)
            generation_times.append(generation_time)
        
        # Analyze generation quality
        quality_metrics = self._analyze_generation_quality(generated_texts)
        
        return {
            'generated_texts': generated_texts,
            'generation_times': generation_times,
            'avg_generation_time': np.mean(generation_times),
            'quality_metrics': quality_metrics
        }
    
    def _generate_text(self, prompt: str, max_length: int, temperature: float) -> str:
        """Generate text from prompt"""
        # Tokenize prompt
        tokens = self.tokenizer.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        
        generated = tokens.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits = self.model(generated)
                
                # Get next token probabilities
                next_token_logits = logits[0, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
        
        # Decode generated text
        generated_text = self.tokenizer.decode(generated[0].tolist())
        return generated_text[len(prompt):]
    
    def _analyze_generation_quality(self, generated_texts: List[str]) -> Dict[str, float]:
        """Analyze quality of generated texts"""
        metrics = {}
        
        # Length analysis
        lengths = [len(text.split()) for text in generated_texts]
        metrics['avg_length'] = np.mean(lengths)
        metrics['length_std'] = np.std(lengths)
        
        # Repetition analysis
        repetition_scores = []
        for text in generated_texts:
            words = text.split()
            if len(words) > 1:
                unique_words = len(set(words))
                repetition_score = 1.0 - (unique_words / len(words))
                repetition_scores.append(repetition_score)
            else:
                repetition_scores.append(0.0)
        
        metrics['avg_repetition'] = np.mean(repetition_scores)
        
        # Coherence analysis (simple)
        coherence_scores = []
        for text in generated_texts:
            # Simple coherence: check for sentence structure
            sentences = text.split('.')
            if len(sentences) > 1:
                # Check if sentences have reasonable length
                sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
                if sentence_lengths:
                    avg_sentence_length = np.mean(sentence_lengths)
                    # Reasonable sentence length (3-20 words)
                    coherence = 1.0 - abs(avg_sentence_length - 10) / 10
                    coherence = max(0, min(1, coherence))
                    coherence_scores.append(coherence)
                else:
                    coherence_scores.append(0.0)
            else:
                coherence_scores.append(0.0)
        
        metrics['avg_coherence'] = np.mean(coherence_scores)
        
        return metrics
    
    def evaluate_consciousness_metrics(self, texts: List[str]) -> Dict[str, Any]:
        """Evaluate consciousness-related metrics"""
        print("🧠 Evaluating consciousness metrics...")
        
        consciousness_scores = []
        attention_patterns = []
        memory_usage = []
        
        with torch.no_grad():
            for text in tqdm(texts, desc="Analyzing consciousness"):
                # Tokenize text
                tokens = self.tokenizer.encode(text, max_length=256)
                if len(tokens) < 2:
                    continue
                
                input_tokens = torch.tensor(tokens[:-1], dtype=torch.long).unsqueeze(0).to(self.device)
                
                # Get consciousness state
                consciousness_state = self.model.get_consciousness_state(input_tokens)
                memory_stats = self.model.get_memory_stats()
                
                # Extract metrics
                consciousness_weights = consciousness_state['consciousness_weights']
                consciousness_scores.append(consciousness_weights.mean().item())
                attention_patterns.append(consciousness_weights.std().item())
                memory_usage.append(memory_stats['learning_efficiency'])
        
        return {
            'avg_consciousness_awareness': np.mean(consciousness_scores),
            'consciousness_std': np.std(consciousness_scores),
            'avg_attention_focus': np.mean(attention_patterns),
            'attention_std': np.std(attention_patterns),
            'avg_memory_usage': np.mean(memory_usage),
            'memory_std': np.std(memory_usage)
        }
    
    def compare_with_baseline(self, texts: List[str]) -> Dict[str, Any]:
        """Compare with baseline models (if available)"""
        print("📊 Comparing with baseline models...")
        
        # Get our model's perplexity
        our_metrics = self.evaluate_perplexity(texts)
        
        # Try to load baseline models for comparison
        baseline_comparisons = {}
        
        # GPT-2 baseline (if available)
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
            gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            gpt2_model.to(self.device)
            gpt2_model.eval()
            
            print("🔄 Computing GPT-2 baseline...")
            gpt2_metrics = self._compute_baseline_perplexity(
                gpt2_model, gpt2_tokenizer, texts
            )
            baseline_comparisons['gpt2'] = gpt2_metrics
            
        except ImportError:
            print("⚠️ Transformers not available for baseline comparison")
        
        return {
            'our_model': our_metrics,
            'baselines': baseline_comparisons
        }
    
    def _compute_baseline_perplexity(self, model, tokenizer, texts: List[str]) -> Dict[str, float]:
        """Compute perplexity for baseline model"""
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for text in tqdm(texts, desc="Baseline perplexity"):
                # Tokenize with baseline tokenizer
                tokens = tokenizer.encode(text, max_length=512, truncation=True)
                
                if len(tokens) < 2:
                    continue
                
                # Create input and target
                input_tokens = torch.tensor(tokens[:-1], dtype=torch.long).unsqueeze(0).to(self.device)
                target_tokens = torch.tensor(tokens[1:], dtype=torch.long).unsqueeze(0).to(self.device)
                
                # Forward pass
                outputs = model(input_tokens)
                logits = outputs.logits
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target_tokens.view(-1),
                    reduction='sum'
                )
                
                total_loss += loss.item()
                total_tokens += len(tokens) - 1
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'perplexity': perplexity,
            'average_loss': avg_loss,
            'total_tokens': total_tokens
        }
    
    def create_evaluation_report(self, output_dir: str = "evaluation_results"):
        """Create comprehensive evaluation report"""
        print("📋 Creating evaluation report...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Test texts
        test_texts = [
            "The brain-inspired language model uses consciousness mechanisms to process information.",
            "Memory consolidation and retrieval are key components of human-like learning.",
            "Spiking neurons provide event-driven processing similar to biological systems.",
            "Hebbian learning rules enable neurons to strengthen connections through co-activation.",
            "Short-term and long-term memory systems work together for effective information storage.",
            "Developmental plasticity allows neural networks to adapt and grow over time.",
            "Minimal data learning enables systems to learn from very few examples.",
            "Consciousness awareness helps focus attention on important information.",
            "Biologically plausible learning avoids backpropagation for more realistic learning.",
            "Event-driven processing is more efficient than traditional continuous processing."
        ]
        
        # Test prompts for generation
        test_prompts = [
            "The brain-inspired language model",
            "Artificial intelligence is revolutionizing",
            "Memory consolidation helps",
            "Consciousness mechanisms enable",
            "Spiking neurons provide"
        ]
        
        # Run evaluations
        print("\n🔍 Running comprehensive evaluation...")
        
        # Perplexity evaluation
        perplexity_results = self.evaluate_perplexity(test_texts)
        
        # Generation quality evaluation
        generation_results = self.evaluate_generation_quality(test_prompts)
        
        # Consciousness metrics evaluation
        consciousness_results = self.evaluate_consciousness_metrics(test_texts)
        
        # Baseline comparison
        comparison_results = self.compare_with_baseline(test_texts)
        
        # Compile results
        evaluation_report = {
            'model_info': {
                'model_path': str(self.model_path),
                'model_config': self.model_config,
                'training_config': self.training_config,
                'checkpoint_info': {
                    'epoch': self.checkpoint.get('epoch', 'unknown'),
                    'step': self.checkpoint.get('step', 'unknown'),
                    'best_val_loss': self.checkpoint.get('best_val_loss', 'unknown')
                }
            },
            'perplexity_evaluation': perplexity_results,
            'generation_evaluation': generation_results,
            'consciousness_evaluation': consciousness_results,
            'baseline_comparison': comparison_results,
            'evaluation_timestamp': time.time()
        }
        
        # Save results
        results_path = output_dir / "evaluation_report.json"
        with open(results_path, 'w') as f:
            json.dump(evaluation_report, f, indent=2, default=str)
        
        # Create summary
        self._create_summary_report(evaluation_report, output_dir)
        
        # Create visualizations
        self._create_visualizations(evaluation_report, output_dir)
        
        print(f"✅ Evaluation report saved to: {results_path}")
        return evaluation_report
    
    def _create_summary_report(self, report: Dict, output_dir: Path):
        """Create human-readable summary report"""
        summary_path = output_dir / "evaluation_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write("BRAIN-INSPIRED LLM EVALUATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Model info
            f.write("MODEL INFORMATION:\n")
            f.write(f"  Model Path: {report['model_info']['model_path']}\n")
            f.write(f"  Dimensions: {report['model_info']['model_config']['dim']}\n")
            f.write(f"  Layers: {report['model_info']['model_config']['num_layers']}\n")
            f.write(f"  Vocab Size: {report['model_info']['model_config']['vocab_size']}\n")
            f.write(f"  Training Epochs: {report['model_info']['checkpoint_info']['epoch']}\n")
            f.write(f"  Best Val Loss: {report['model_info']['checkpoint_info']['best_val_loss']:.4f}\n\n")
            
            # Perplexity results
            f.write("PERPLEXITY EVALUATION:\n")
            f.write(f"  Overall Perplexity: {report['perplexity_evaluation']['overall_perplexity']:.2f}\n")
            f.write(f"  Average Loss: {report['perplexity_evaluation']['average_loss']:.4f}\n")
            f.write(f"  Total Tokens: {report['perplexity_evaluation']['total_tokens']:,}\n")
            f.write(f"  Perplexity Std: {report['perplexity_evaluation']['perplexity_std']:.2f}\n\n")
            
            # Generation results
            f.write("GENERATION EVALUATION:\n")
            f.write(f"  Average Generation Time: {report['generation_evaluation']['avg_generation_time']:.3f}s\n")
            f.write(f"  Average Length: {report['generation_evaluation']['quality_metrics']['avg_length']:.1f} words\n")
            f.write(f"  Average Repetition: {report['generation_evaluation']['quality_metrics']['avg_repetition']:.3f}\n")
            f.write(f"  Average Coherence: {report['generation_evaluation']['quality_metrics']['avg_coherence']:.3f}\n\n")
            
            # Consciousness results
            f.write("CONSCIOUSNESS EVALUATION:\n")
            f.write(f"  Average Consciousness Awareness: {report['consciousness_evaluation']['avg_consciousness_awareness']:.4f}\n")
            f.write(f"  Average Attention Focus: {report['consciousness_evaluation']['avg_attention_focus']:.4f}\n")
            f.write(f"  Average Memory Usage: {report['consciousness_evaluation']['avg_memory_usage']:.4f}\n\n")
            
            # Baseline comparison
            if report['baseline_comparison']['baselines']:
                f.write("BASELINE COMPARISON:\n")
                for baseline_name, baseline_metrics in report['baseline_comparison']['baselines'].items():
                    f.write(f"  {baseline_name.upper()} Perplexity: {baseline_metrics['perplexity']:.2f}\n")
                    our_ppl = report['baseline_comparison']['our_model']['overall_perplexity']
                    baseline_ppl = baseline_metrics['perplexity']
                    improvement = ((baseline_ppl - our_ppl) / baseline_ppl) * 100
                    f.write(f"  Improvement over {baseline_name}: {improvement:.1f}%\n")
                f.write("\n")
            
            # Generated examples
            f.write("GENERATED EXAMPLES:\n")
            for i, (prompt, generated) in enumerate(zip(
                ["The brain-inspired language model", "Artificial intelligence is revolutionizing"],
                report['generation_evaluation']['generated_texts'][:2]
            )):
                f.write(f"  Example {i+1}:\n")
                f.write(f"    Prompt: {prompt}\n")
                f.write(f"    Generated: {generated}\n\n")
        
        print(f"📋 Summary report saved to: {summary_path}")
    
    def _create_visualizations(self, report: Dict, output_dir: Path):
        """Create visualization plots"""
        if not VISUALIZATION_AVAILABLE:
            print("⚠️ Visualization libraries not available, skipping plots")
            return
            
        try:
            # Set style
            try:
                plt.style.use('seaborn-v0_8')
            except:
                plt.style.use('default')
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Brain-Inspired LLM Evaluation Results', fontsize=16)
            
            # 1. Perplexity distribution
            if 'perplexity_evaluation' in report:
                ppl_data = [report['perplexity_evaluation']['overall_perplexity']]
                if report['baseline_comparison']['baselines']:
                    for baseline_metrics in report['baseline_comparison']['baselines'].values():
                        ppl_data.append(baseline_metrics['perplexity'])
                
                axes[0, 0].bar(['Our Model'] + list(report['baseline_comparison']['baselines'].keys()), ppl_data)
                axes[0, 0].set_title('Perplexity Comparison')
                axes[0, 0].set_ylabel('Perplexity')
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. Consciousness metrics
            if 'consciousness_evaluation' in report:
                consciousness_data = [
                    report['consciousness_evaluation']['avg_consciousness_awareness'],
                    report['consciousness_evaluation']['avg_attention_focus'],
                    report['consciousness_evaluation']['avg_memory_usage']
                ]
                axes[0, 1].bar(['Awareness', 'Attention', 'Memory'], consciousness_data)
                axes[0, 1].set_title('Consciousness Metrics')
                axes[0, 1].set_ylabel('Score')
            
            # 3. Generation quality
            if 'generation_evaluation' in report:
                quality_data = [
                    report['generation_evaluation']['quality_metrics']['avg_length'],
                    report['generation_evaluation']['quality_metrics']['avg_coherence'] * 10,  # Scale for visibility
                    report['generation_evaluation']['quality_metrics']['avg_repetition'] * 10
                ]
                axes[1, 0].bar(['Length', 'Coherence×10', 'Repetition×10'], quality_data)
                axes[1, 0].set_title('Generation Quality Metrics')
                axes[1, 0].set_ylabel('Score')
            
            # 4. Training history (if available)
            if 'training_history' in report.get('model_info', {}).get('checkpoint_info', {}):
                # This would require loading training history from checkpoint
                axes[1, 1].text(0.5, 0.5, 'Training History\n(Not Available)', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Training History')
            else:
                axes[1, 1].text(0.5, 0.5, 'Training History\n(Not Available)', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Training History')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = output_dir / "evaluation_plots.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Visualization saved to: {plot_path}")
            
        except ImportError:
            print("⚠️ Matplotlib not available, skipping visualizations")
        except Exception as e:
            print(f"⚠️ Failed to create visualizations: {e}")

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate Brain-Inspired LLM')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for evaluation results')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use for evaluation')
    
    args = parser.parse_args()
    
    print("🔍 BRAIN-INSPIRED LLM EVALUATION")
    print("=" * 50)
    print(f"📊 Model: {args.model_path}")
    print(f"💾 Output: {args.output_dir}")
    print(f"🖥️ Device: {args.device}")
    print("=" * 50)
    
    # Create evaluator
    evaluator = ProductionEvaluator(args.model_path, args.device)
    
    # Run evaluation
    report = evaluator.create_evaluation_report(args.output_dir)
    
    print("\n✅ Evaluation completed!")
    print(f"📋 Report saved to: {args.output_dir}")
    
    return report

if __name__ == "__main__":
    report = main()
