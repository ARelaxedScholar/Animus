#!/usr/bin/env python3
"""
DSPy Bridge for Animus - Phase 1: Judge Only

This bridge exposes the DSPy-optimized Judge to the Rust orchestrator.
The Writer remains in Orichalcum with hardcoded prompts for now.

The Judge learns to predict real-world performance (views, retention, likes)
from historical script → performance_score data.

Usage:
    # From Rust via stdin/stdout JSON (same pattern as moviepy_bridge)
    echo '{"action": "evaluate", "script": {...}, "topic_brief": {...}}' | python dspy_bridge.py
    
    # Compile/optimize the Judge from training data
    python dspy_bridge.py --compile --training-data path/to/data.jsonl
"""

import json
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    print("Warning: DSPy not installed. Run: pip install dspy-ai", file=sys.stderr)


# =============================================================================
# Configuration
# =============================================================================

# Where compiled programs are stored
COMPILED_PROGRAMS_DIR = Path(__file__).parent.parent.parent / "models" / "dspy"

# Default model for the Judge (can be overridden via env)
JUDGE_MODEL = os.getenv("DSPY_JUDGE_MODEL", "deepseek/deepseek-chat")

# Fallback to hardcoded prompt if no compiled program exists
USE_FALLBACK = os.getenv("DSPY_USE_FALLBACK", "true").lower() == "true"


# =============================================================================
# DSPy Signatures
# =============================================================================

if DSPY_AVAILABLE:
    class ScriptPerformancePredictor(dspy.Signature):
        """Predict how well a YouTube script will perform based on real-world metrics.
        
        You are evaluating scripts for a wisdom/self-improvement YouTube channel.
        Your predictions should correlate with actual viewer behavior:
        - High retention = viewers watch most of the video
        - High engagement = viewers like and comment
        - Viral potential = viewers share and return
        
        Base your prediction on patterns you've learned from scripts that actually
        performed well vs poorly in the real world.
        """
        
        topic: str = dspy.InputField(desc="The video topic/title concept")
        target_duration_minutes: int = dspy.InputField(desc="Target video length")
        hook_angle: str = dspy.InputField(desc="The hook/angle for grabbing attention")
        script_text: str = dspy.InputField(desc="The full script narration text")
        word_count: int = dspy.InputField(desc="Total word count of the script")
        
        predicted_score: float = dspy.OutputField(
            desc="Predicted performance score from 0.0 to 1.0, where 1.0 means "
                 "exceptional real-world performance (high retention, likes, views)"
        )
        confidence: float = dspy.OutputField(
            desc="Your confidence in this prediction from 0.0 to 1.0"
        )
        reasoning: str = dspy.OutputField(
            desc="Brief explanation of why this script will perform at this level"
        )
        
        # Detailed breakdown (matches existing evaluation structure)
        hook_strength: float = dspy.OutputField(
            desc="Hook effectiveness 1-10: Does the opening create urgency to keep watching?"
        )
        pacing_retention: float = dspy.OutputField(
            desc="Pacing score 1-10: Does the rhythm vary? Are there retention hooks?"
        )
        authenticity: float = dspy.OutputField(
            desc="Authenticity score 1-10: Does this sound human or AI-generated?"
        )
        
        strengths: str = dspy.OutputField(
            desc="Comma-separated list of script strengths"
        )
        weaknesses: str = dspy.OutputField(
            desc="Comma-separated list of script weaknesses"
        )
        improvement_suggestions: str = dspy.OutputField(
            desc="Specific suggestions for improving the script"
        )


    class JudgeModule(dspy.Module):
        """The Judge module that predicts script performance.
        
        This module can be optimized via DSPy's BootstrapFewShot to learn
        from real (script, performance_score) pairs.
        """
        
        def __init__(self):
            super().__init__()
            self.predictor = dspy.ChainOfThought(ScriptPerformancePredictor)
        
        def forward(
            self,
            topic: str,
            target_duration_minutes: int,
            hook_angle: str,
            script_text: str,
        ) -> dspy.Prediction:
            word_count = len(script_text.split())
            
            return self.predictor(
                topic=topic,
                target_duration_minutes=target_duration_minutes,
                hook_angle=hook_angle,
                script_text=script_text,
                word_count=word_count,
            )


# =============================================================================
# Judge Manager (handles loading/saving compiled programs)
# =============================================================================

class JudgeManager:
    """Manages the DSPy Judge module, including compilation and inference."""
    
    def __init__(self):
        self.module: Optional[Any] = None
        self.is_compiled = False
        self._setup_model()
    
    def _setup_model(self):
        """Configure the LLM backend for DSPy."""
        if not DSPY_AVAILABLE:
            return
        
        # Configure DSPy with the judge model
        # Support multiple providers via litellm-style model strings
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        if "deepseek" in JUDGE_MODEL.lower():
            lm = dspy.LM(
                model=JUDGE_MODEL,
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                api_base="https://api.deepseek.com/v1",
            )
        elif "gemini" in JUDGE_MODEL.lower():
            lm = dspy.LM(
                model=JUDGE_MODEL,
                api_key=os.getenv("GEMINI_API_KEY"),
            )
        else:
            # Default to OpenAI-compatible
            lm = dspy.LM(
                model=JUDGE_MODEL,
                api_key=api_key,
            )
        
        dspy.configure(lm=lm)
    
    def load_or_create(self) -> bool:
        """Load a compiled program or create a fresh one."""
        if not DSPY_AVAILABLE:
            return False
        
        compiled_path = COMPILED_PROGRAMS_DIR / "judge_compiled.json"
        
        if compiled_path.exists():
            try:
                self.module = JudgeModule()
                self.module.load(str(compiled_path))
                self.is_compiled = True
                print(f"Loaded compiled Judge from {compiled_path}", file=sys.stderr)
                return True
            except Exception as e:
                print(f"Failed to load compiled program: {e}", file=sys.stderr)
        
        # Create fresh module (unoptimized)
        self.module = JudgeModule()
        self.is_compiled = False
        print("Using unoptimized Judge (no compiled program found)", file=sys.stderr)
        return True
    
    def evaluate(self, script: Dict[str, Any], topic_brief: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a script and return structured feedback."""
        if not DSPY_AVAILABLE or not self.module:
            return self._fallback_evaluate(script, topic_brief)
        
        try:
            # Extract script text
            script_text = script.get("full_text", "")
            if not script_text:
                # Reconstruct from sections
                parts = []
                if hook := script.get("hook", {}).get("narration"):
                    parts.append(hook)
                for section in script.get("sections", []):
                    if narration := section.get("narration"):
                        parts.append(narration)
                if cta := script.get("cta", {}).get("narration"):
                    parts.append(cta)
                script_text = "\n\n".join(parts)
            
            # Run the Judge
            result = self.module(
                topic=topic_brief.get("topic", "Unknown"),
                target_duration_minutes=topic_brief.get("target_duration_minutes", 15),
                hook_angle=topic_brief.get("hook_angle", ""),
                script_text=script_text,
            )
            
            # Parse outputs (handle both string and numeric returns)
            def safe_float(val, default=5.0):
                if isinstance(val, (int, float)):
                    return float(val)
                try:
                    return float(str(val).strip())
                except:
                    return default
            
            def safe_list(val):
                if isinstance(val, list):
                    return val
                if isinstance(val, str):
                    return [s.strip() for s in val.split(",") if s.strip()]
                return []
            
            # Convert to legacy evaluation format
            overall_score = safe_float(result.predicted_score, 0.5) * 10  # Scale to 0-10
            
            return {
                "overall_score": overall_score,
                "criteria": {
                    "hook_strength": {
                        "score": safe_float(result.hook_strength, 5.0),
                        "notes": ""
                    },
                    "pacing_retention": {
                        "score": safe_float(result.pacing_retention, 5.0),
                        "notes": ""
                    },
                    "authenticity": {
                        "score": safe_float(result.authenticity, 5.0),
                        "notes": ""
                    },
                    "wisdom_integration": {"score": 5.0, "notes": ""},
                    "duration_accuracy": {"score": 5.0, "notes": ""},
                    "cta_quality": {"score": 5.0, "notes": ""},
                    "ai_detection": {
                        "score": safe_float(result.authenticity, 5.0),
                        "notes": ""
                    },
                },
                "strengths": safe_list(result.strengths),
                "weaknesses": safe_list(result.weaknesses),
                "ai_telltale_signs": [],
                "specific_improvements": [
                    {
                        "location": "general",
                        "issue": "See suggestions",
                        "suggestion": str(result.improvement_suggestions)
                    }
                ] if result.improvement_suggestions else [],
                "dspy_metadata": {
                    "predicted_score": safe_float(result.predicted_score, 0.5),
                    "confidence": safe_float(result.confidence, 0.5),
                    "reasoning": str(result.reasoning),
                    "is_compiled": self.is_compiled,
                }
            }
            
        except Exception as e:
            print(f"DSPy evaluation failed: {e}", file=sys.stderr)
            if USE_FALLBACK:
                return self._fallback_evaluate(script, topic_brief)
            raise
    
    def _fallback_evaluate(self, script: Dict[str, Any], topic_brief: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback evaluation when DSPy is unavailable."""
        # Return a neutral evaluation that doesn't block the pipeline
        return {
            "overall_score": 6.0,
            "criteria": {
                "hook_strength": {"score": 6.0, "notes": "Fallback evaluation"},
                "pacing_retention": {"score": 6.0, "notes": ""},
                "wisdom_integration": {"score": 6.0, "notes": ""},
                "authenticity": {"score": 6.0, "notes": ""},
                "duration_accuracy": {"score": 6.0, "notes": ""},
                "cta_quality": {"score": 6.0, "notes": ""},
                "ai_detection": {"score": 6.0, "notes": ""},
            },
            "strengths": ["Fallback mode - no detailed analysis"],
            "weaknesses": ["DSPy unavailable - using fallback"],
            "ai_telltale_signs": [],
            "specific_improvements": [],
            "dspy_metadata": {
                "is_fallback": True,
                "reason": "DSPy not available or module not loaded"
            }
        }
    
    def compile(self, training_data: List[Dict[str, Any]], output_path: Optional[Path] = None):
        """Compile/optimize the Judge from training data.
        
        Training data format:
        [
            {
                "topic_brief": {...},
                "script": {...},
                "performance_score": 0.75  # The real-world score (0-1)
            },
            ...
        ]
        """
        if not DSPY_AVAILABLE:
            raise RuntimeError("DSPy not installed")
        
        print(f"Compiling Judge from {len(training_data)} examples...", file=sys.stderr)
        
        # Create training examples
        trainset = []
        for item in training_data:
            topic_brief = item.get("topic_brief", {})
            script = item.get("script", {})
            score = item.get("performance_score", 0.5)
            
            # Extract script text
            script_text = script.get("full_text", "")
            if not script_text:
                parts = []
                if hook := script.get("hook", {}).get("narration"):
                    parts.append(hook)
                for section in script.get("sections", []):
                    if narration := section.get("narration"):
                        parts.append(narration)
                if cta := script.get("cta", {}).get("narration"):
                    parts.append(cta)
                script_text = "\n\n".join(parts)
            
            example = dspy.Example(
                topic=topic_brief.get("topic", "Unknown"),
                target_duration_minutes=topic_brief.get("target_duration_minutes", 15),
                hook_angle=topic_brief.get("hook_angle", ""),
                script_text=script_text,
                word_count=len(script_text.split()),
                # Target outputs
                predicted_score=score,
            ).with_inputs("topic", "target_duration_minutes", "hook_angle", "script_text", "word_count")
            
            trainset.append(example)
        
        # Define metric: how close is the prediction to actual performance?
        def prediction_accuracy(example, prediction, trace=None):
            predicted = float(prediction.predicted_score)
            actual = float(example.predicted_score)
            # Score based on how close the prediction is (1.0 = perfect)
            error = abs(predicted - actual)
            return max(0, 1.0 - error)
        
        # Use BootstrapFewShot for optimization
        from dspy.teleprompt import BootstrapFewShot
        
        optimizer = BootstrapFewShot(
            metric=prediction_accuracy,
            max_bootstrapped_demos=4,
            max_labeled_demos=8,
        )
        
        # Compile
        self.module = JudgeModule()
        compiled_module = optimizer.compile(
            self.module,
            trainset=trainset,
        )
        
        # Save
        output_path = output_path or (COMPILED_PROGRAMS_DIR / "judge_compiled.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        compiled_module.save(str(output_path))
        
        self.module = compiled_module
        self.is_compiled = True
        
        print(f"Judge compiled and saved to {output_path}", file=sys.stderr)
        return str(output_path)


# =============================================================================
# Main Entry Point (stdin/stdout JSON interface)
# =============================================================================

def main():
    """Read JSON from stdin, process, write JSON to stdout."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DSPy Bridge for Animus Judge")
    parser.add_argument("--compile", action="store_true", help="Compile the Judge from training data")
    parser.add_argument("--training-data", type=str, help="Path to training data JSONL file")
    parser.add_argument("--output", type=str, help="Output path for compiled program")
    args = parser.parse_args()
    
    # Compilation mode
    if args.compile:
        if not args.training_data:
            print("Error: --training-data required for compilation", file=sys.stderr)
            sys.exit(1)
        
        # Load training data
        training_data = []
        with open(args.training_data, 'r') as f:
            for line in f:
                if line.strip():
                    training_data.append(json.loads(line))
        
        if len(training_data) < 5:
            print(f"Warning: Only {len(training_data)} training examples. Recommend 10+ for reliable optimization.", file=sys.stderr)
        
        manager = JudgeManager()
        output_path = Path(args.output) if args.output else None
        result_path = manager.compile(training_data, output_path)
        
        print(json.dumps({"success": True, "compiled_path": result_path}))
        return
    
    # Inference mode (stdin/stdout)
    true_stdout = sys.stdout
    sys.stdout = sys.stderr  # Redirect prints to stderr
    
    try:
        input_data = sys.stdin.read()
        if not input_data:
            print(json.dumps({"success": False, "error": "Empty input"}), file=true_stdout)
            sys.exit(1)
        
        config = json.loads(input_data)
        action = config.get("action", "evaluate")
        
        manager = JudgeManager()
        manager.load_or_create()
        
        if action == "evaluate":
            script = config.get("script", {})
            topic_brief = config.get("topic_brief", {})
            
            result = manager.evaluate(script, topic_brief)
            print(json.dumps({"success": True, "evaluation": result}), file=true_stdout)
        
        elif action == "health":
            print(json.dumps({
                "success": True,
                "dspy_available": DSPY_AVAILABLE,
                "is_compiled": manager.is_compiled,
                "model": JUDGE_MODEL,
            }), file=true_stdout)
        
        else:
            print(json.dumps({"success": False, "error": f"Unknown action: {action}"}), file=true_stdout)
    
    except json.JSONDecodeError as e:
        print(json.dumps({"success": False, "error": f"Invalid JSON: {e}"}), file=true_stdout)
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}), file=true_stdout)
        sys.exit(1)
    finally:
        true_stdout.flush()


if __name__ == "__main__":
    main()
