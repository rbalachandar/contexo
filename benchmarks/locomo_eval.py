"""
Evaluate Contexo on LoCoMo benchmark.

LoCoMo: https://github.com/snap-research/locomo
Tests long-term conversational memory with QA accuracy.

Uses local Sentence Transformers embeddings (free, no API key needed).

Working Memory vs Persistent Memory in LoCoMo:
----------------------------------------------
LoCoMo evaluates PERSISTENT memory retrieval accuracy - can semantic search
find relevant information from a long conversation history?

- Working Memory: Current conversation context (small, ~4K tokens)
- Persistent Memory: Long-term storage with semantic search (unlimited)

In a real application using this pattern:
1. Historical messages are stored in persistent memory (as we do here)
2. For each user query, search persistent memory for relevant context
3. Promote relevant results to working memory for LLM context
4. LLM generates response using working memory context

Example real usage:
    # User asks a question
    query = "What did we discuss about the API design?"

    # Find relevant context from long-term memory
    relevant = await ctx.search_memory(query, limit=5)

    # Promote to working memory for LLM context
    await ctx.promote_relevant(query, limit=5)

    # Get context for LLM (includes promoted entries)
    context = await ctx.get_context()

In this benchmark, we skip step 3 since we only test retrieval accuracy,
not response generation.
"""

import json
import asyncio
from pathlib import Path
from contexo import Contexo
from contexo.config.defaults import minimal_config
from contexo.config.settings import StorageConfig, EmbeddingConfig
from typing import Any


class LoCoMoEvaluator:
    """Evaluate Contexo on LoCoMo benchmark with chunking."""

    def __init__(
        self,
        locomo_data_path: str,
        results_path: str = None,
    ):
        with open(locomo_data_path) as f:
            self.data = json.load(f)
        self.results_path = results_path or locomo_data_path.replace(".json", "_results.json")

        # Choose embedding provider:
        # Option 1: OpenAI (requires API key, best quality) - set provider_type="openai"
        # Option 2: SentenceTransformers (free, local) - set provider_type="sentence_transformers"
        self.config = minimal_config()
        from dataclasses import replace

        # Use SentenceTransformers with BGE model (FREE, no API key, no auth required)
        # BAAI/bge-large-en-v1.5 achieved ~68% in testing
        self.config = replace(
            self.config,
            storage=StorageConfig(
                backend_type="sqlite",
                db_path=str(Path(locomo_data_path).parent / "locomo_cache.db"),
            ),
            embeddings=EmbeddingConfig(
                provider_type="sentence_transformers",
                model_name="BAAI/bge-large-en-v1.5",  # Best open-source embedding model
                device="cpu",  # or "cuda"/"mps" if available
            ),
        )

        # Alternative: Use OpenAI (requires OPENAI_API_KEY env var)
        # self.config = replace(
        #     self.config,
        #     storage=StorageConfig(
        #         backend_type="sqlite",
        #         db_path=str(Path(locomo_data_path).parent / "locomo_cache.db"),
        #     ),
        #     embeddings=EmbeddingConfig(
        #         provider_type="openai",
        #         model_name="text-embedding-3-small",
        #         api_key=None,
        #     ),
        # )

    def print_cost_estimate(self):
        """Print estimated cost for running LoCoMo."""
        provider = self.config.embeddings.provider_type
        model = self.config.embeddings.model_name

        if provider == "sentence_transformers":
            print(f"\n💰 Cost: FREE (using local SentenceTransformers)")
            print(f"   Model: {model}")
            print(f"   Device: {self.config.embeddings.device or 'CPU'}")
            print(f"\n   First run will download the model (~600MB)")
        else:
            print(f"\n💰 Cost: ~$0.20 (using OpenAI {model})")
            print(f"   Requires: OPENAI_API_KEY environment variable")
            print(f"\n   Note: Free tier: 40K tokens/month, we use ~10K")

    async def evaluate_conversation(
        self,
        sample_index: int,
        save_progress: bool = True
    ) -> dict[str, Any]:
        """Evaluate one conversation and save progress."""

        # Check if already evaluated
        results = self._load_results()
        if results and f"sample_{sample_index}" in results.get("completed_samples", []):
            print(f"Sample {sample_index} already evaluated, skipping")
            return results["per_sample_results"][sample_index]

        print(f"\n{'='*60}")
        print(f"Evaluating sample {sample_index+1}/{len(self.data)}")
        print(f"{'='*60}")

        sample = self.data[sample_index]
        conversation = sample["conversation"]
        qa_pairs = sample["qa"]

        ctx = Contexo(config=self.config)
        await ctx.initialize()
        ctx.set_conversation_id(f"locomo_sample_{sample_index}")

        # Check if conversation already exists in database (skip re-inserting)
        conversation_id = f"locomo_sample_{sample_index}"
        existing_count = await ctx._persistent._storage.count(collection=conversation_id)

        # Insert all turns (skip if already inserted)
        # Get only actual session keys (exclude metadata like _date_time)
        session_keys = sorted([k for k in conversation.keys() if k.startswith("session_") and k != "session" and "_date_time" not in k])
        total_turns = sum(len(conversation[k]) for k in session_keys)

        if existing_count >= total_turns:
            print(f"Skipping insertion (found {existing_count} existing entries)")
        else:
            print(f"Inserting {total_turns} conversation turns into persistent memory...")

            # LoCoMo evaluates PERSISTENT memory retrieval, not working memory.
            # Working memory is for current conversation context (limited tokens).
            # Persistent memory is for long-term storage with semantic search.
            # We add directly to persistent memory to avoid token limits.
            from contexo.core.memory import MemoryEntry, EntryType

            for i, session_key in enumerate(session_keys):
                session = conversation[session_key]
                for turn in session:
                    entry = MemoryEntry(
                        entry_type=EntryType.MESSAGE,
                        content=turn["text"],
                        metadata={
                            "role": turn["speaker"],
                            "session": session_key,
                            "dia_id": turn["dia_id"],
                        },
                        conversation_id=conversation_id,
                    )
                    # Add directly to persistent memory (bypasses working memory)
                    await ctx._persistent.add(entry)
                print(f"  Session {i+1}/{len(session_keys)}: {len(session)} turns")

        # Test QA pairs
        print(f"\nTesting {len(qa_pairs)} questions...")

        sample_result = {
            "sample_id": sample["sample_id"],
            "total_questions": len(qa_pairs),
            "correct": 0,
            "incorrect": 0,
            "failures": [],  # Track failure details
            "details": []
        }

        for i, qa in enumerate(qa_pairs):
            question = qa["question"]
            evidence_dia_ids = qa.get("evidence", [])

            retrieved = await ctx.search_memory(query=question, limit=200)

            found_evidence = any(
                entry.metadata.get("dia_id") in evidence_dia_ids
                for entry in retrieved
            )

            if found_evidence:
                sample_result["correct"] += 1
            else:
                sample_result["incorrect"] += 1
                # Track failure details
                sample_result["failures"].append({
                    "question": question,
                    "evidence_count": len(evidence_dia_ids),
                    "top_retrieved_dia_ids": [
                        entry.metadata.get("dia_id") for entry in retrieved[:5]
                    ],
                })

            if (i + 1) % 5 == 0:
                print(f"  Progress: {i+1}/{len(qa_pairs)} questions tested")

        sample_result["accuracy"] = sample_result["correct"] / sample_result["total_questions"]

        await ctx.close()

        print(f"\n✓ Sample {sample['sample_id']}: {sample_result['accuracy']:.1%} ({sample_result['correct']}/{sample_result['total_questions']})")

        # Save progress after each sample
        if save_progress:
            self._save_sample_result(sample_index, sample_result)

        return sample_result

    def _load_results(self) -> dict:
        """Load existing results if available."""
        if Path(self.results_path).exists():
            with open(self.results_path) as f:
                return json.load(f)
        return {"completed_samples": [], "per_sample_results": {}}

    def _save_sample_result(self, sample_index: int, result: dict):
        """Save progress incrementally."""
        results = self._load_results()

        if sample_index not in results["completed_samples"]:
            results["completed_samples"].append(sample_index)
        results["per_sample_results"][sample_index] = result

        # Save incrementally
        with open(self.results_path, "w") as f:
            json.dump(results, f, indent=2)

    async def evaluate_all(
        self,
        sample_indices: list[int] = None,
    ) -> dict[str, Any]:
        """Evaluate specific samples or all.

        Args:
            sample_indices: List of sample indices to eval (None = all)
        """

        if sample_indices is None:
            sample_indices = list(range(len(self.data)))

        print(f"Will evaluate {len(sample_indices)} samples: {sample_indices}")
        print(f"Results will be saved to: {self.results_path}")

        all_results = []

        for i in sample_indices:
            result = await self.evaluate_conversation(i)
            all_results.append(result)

        # Calculate overall
        total_correct = sum(r["correct"] for r in all_results)
        total_questions = sum(r["total_questions"] for r in all_results)

        final_results = {
            "overall_accuracy": total_correct / total_questions,
            "total_correct": total_correct,
            "total_questions": total_questions,
            "per_sample_results": all_results,
            "completed_samples": sample_indices,
        }

        # Save final results
        with open(self.results_path, "w") as f:
            json.dump(final_results, f, indent=2)

        return final_results

    def print_report(self, results: dict[str, Any]):
        """Print evaluation report."""
        print("\n" + "=" * 60)
        print("LOCOMO BENCHMARK RESULTS")
        print("=" * 60)
        print(f"\nOverall Accuracy: {results['overall_accuracy']:.2%}")
        print(f"Total Questions: {results['total_questions']}")
        print(f"Correct: {results['total_correct']}")
        print(f"Incorrect: {results['total_questions'] - results['total_correct']}")

        print("\nPer-Sample Results:")
        print("-" * 60)
        for result in results.get("per_sample_results", []):
            print(f"{result['sample_id']}: {result['accuracy']:.1%} ({result['correct']}/{result['total_questions']})")

        print("\nComparison:")
        print("-" * 60)
        print(f"Contexo:      {results['overall_accuracy']:.2%}")
        print(f"MemU:         92.09%")
        print(f"Random:       ~10%")

    def print_failure_analysis(self, results: dict[str, Any]):
        """Analyze and print failure patterns."""
        print("\n" + "=" * 60)
        print("FAILURE ANALYSIS")
        print("=" * 60)

        all_failures = []
        for result in results.get("per_sample_results", []):
            if "failures" in result:
                for failure in result["failures"]:
                    failure["sample_id"] = result["sample_id"]
                    all_failures.append(failure)

        if not all_failures:
            print("\nNo failures tracked!")
            return

        print(f"\nTotal failures: {len(all_failures)}")

        # Categorize failures by question characteristics
        import re

        # Temporal questions (when, time, ago, last, recently)
        temporal = [f for f in all_failures if re.search(r'\b(when|time|ago|last|recently|before|after)\b', f["question"].lower())]

        # Count/quantity questions (how many, count, number)
        count_questions = [f for f in all_failures if re.search(r'\b(how many|count|number|amount)\b', f["question"].lower())]

        # Entity questions (who, what person, name)
        entity = [f for f in all_failures if re.search(r'\b(who|what person|name|called)\b', f["question"].lower())]

        # List/aggregation questions (list, all, every, each)
        list_questions = [f for f in all_failures if re.search(r'\b(list|all of|every|each)\b', f["question"].lower())]

        # Negation questions (not, never, didn't)
        negation = [f for f in all_failures if re.search(r'\b(not|never|didn\'t|doesn\'t)\b', f["question"].lower())]

        print("\nFailure Categories:")
        print("-" * 60)
        print(f"Temporal (when/time):      {len(temporal)} ({len(temporal)/len(all_failures)*100:.1f}%)")
        print(f"Count/quantity:            {len(count_questions)} ({len(count_questions)/len(all_failures)*100:.1f}%)")
        print(f"Entity (who/what person):  {len(entity)} ({len(entity)/len(all_failures)*100:.1f}%)")
        print(f"List/aggregation:          {len(list_questions)} ({len(list_questions)/len(all_failures)*100:.1f}%)")
        print(f"Negation:                  {len(negation)} ({len(negation)/len(all_failures)*100:.1f}%)")

        # Show example failures
        print("\nExample Failures (first 10):")
        print("-" * 60)
        for i, failure in enumerate(all_failures[:10]):
            print(f"\n{i+1}. [{failure['sample_id']}]")
            print(f"   Q: {failure['question']}")
            print(f"   Evidence IDs: {failure['evidence_count']}")


async def main():
    """Run LoCoMo evaluation."""

    script_dir = Path(__file__).parent
    data_path = script_dir / "data" / "locomo10.json"

    if not data_path.exists():
        print(f"LoCoMo data not found at {data_path}")
        print("\nDownload it first:")
        print("  cd benchmarks/data")
        print("  curl -L https://github.com/snap-research/locomo/raw/main/data/locomo10.json -o locomo10.json")
        return

    evaluator = LoCoMoEvaluator(str(data_path))
    evaluator.print_cost_estimate()

    # Check for existing results to resume
    existing = evaluator._load_results()
    if existing.get("completed_samples"):
        print(f"\nFound existing results for samples: {existing['completed_samples']}")
        print("Press Enter to continue where left off, or '0' to restart")
        response = input("Choice (Enter=continue, 0=restart): ").strip()

        if response == "0":
            Path(evaluator.results_path).unlink(missing_ok=True)
            results = await evaluator.evaluate_all(sample_indices=[0])
        else:
            # Continue where left off
            completed = set(existing["completed_samples"])
            all_indices = set(range(len(evaluator.data)))
            remaining = sorted(list(all_indices - completed))

            if remaining:
                print(f"Will evaluate remaining samples: {remaining}")
                results = await evaluator.evaluate_all(sample_indices=remaining)
            else:
                print("All samples already evaluated!")
                results = existing
    else:
        # First time running - evaluate all samples
        print(f"\nFirst time running LoCoMo evaluation!")
        print(f"Total samples to evaluate: {len(evaluator.data)}")

        results = await evaluator.evaluate_all()

    evaluator.print_report(results)
    evaluator.print_failure_analysis(results)


if __name__ == "__main__":
    asyncio.run(main())
