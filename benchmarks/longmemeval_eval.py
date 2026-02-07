"""
Evaluate Contexo on LongMemEval benchmark.

LongMemEval: https://github.com/xiaowu0162/LongMemEval
Tests long-term interactive memory with 5 core abilities:
- Information Extraction
- Multi-Session Reasoning
- Knowledge Updates
- Temporal Reasoning
- Abstention

Uses local Sentence Transformers embeddings (free, no API key needed).

Dataset Info:
- longmemeval_s_cleaned.json: ~115K tokens, ~40 sessions per question
- longmemeval_m_cleaned.json: ~500 sessions per question
- 500 questions total

Working Memory vs Persistent Memory in LongMemEval:
----------------------------------------------------
LongMemEval evaluates PERSISTENT memory retrieval accuracy - can semantic search
find relevant information from hundreds of chat sessions?

- Working Memory: Current conversation context (small, ~4K tokens)
- Persistent Memory: Long-term storage with semantic search (unlimited)

In a real application using this pattern:
1. Historical sessions are stored in persistent memory (as we do here)
2. For each question, search persistent memory for relevant context
3. Promote relevant results to working memory for LLM context
4. LLM generates response using working memory context

Example real usage:
    # User asks a question
    query = "What did we discuss about the API design?"

    # Find relevant context from long-term memory
    relevant = await ctx.search_memory(query, limit=10)

    # Promote to working memory for LLM context
    await ctx.promote_relevant(query, limit=5)

    # Get context for LLM (includes promoted entries)
    context = await ctx.get_context()

In this benchmark, we skip step 3 since we test retrieval accuracy,
not response generation. The retrieved context can be fed to an LLM
for answer generation.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Optional

from contexo import Contexo
from contexo.config.defaults import minimal_config
from contexo.config.settings import StorageConfig, EmbeddingConfig
from contexo.core.memory import MemoryEntry, EntryType


class LongMemEvalRunner:
    """Evaluate Contexo on LongMemEval benchmark."""

    def __init__(
        self,
        data_path: str,
        results_path: str = None,
        use_llm: bool = False,
    ):
        """Initialize the LongMemEval runner.

        Args:
            data_path: Path to LongMemEval JSON file (longmemeval_s_cleaned.json)
            results_path: Path to save results (default: data_path with _results.jsonl suffix)
            use_llm: If True, use LLM to generate answers (requires OPENAI_API_KEY)
        """
        with open(data_path) as f:
            self.data = json.load(f)

        self.results_path = results_path or data_path.replace(".json", "_contexo_results.jsonl")
        self.use_llm = use_llm

        # Configure Contexo with Sentence Transformers (free, local)
        from dataclasses import replace

        self.config = minimal_config()
        self.config = replace(
            self.config,
            storage=StorageConfig(
                backend_type="sqlite",
                db_path=str(Path(data_path).parent / "longmemeval_cache.db"),
            ),
            embeddings=EmbeddingConfig(
                provider_type="sentence_transformers",
                model_name="BAAI/bge-large-en-v1.5",
                device="cpu",
            ),
        )

        # For LLM-based answer generation
        self._llm_client = None
        if use_llm:
            try:
                from openai import AsyncOpenAI
                api_key = None  # Will use OPENAI_API_KEY env var
                self._llm_client = AsyncOpenAI(api_key=api_key)
            except ImportError:
                print("Warning: openai not installed, LLM generation disabled")
                self.use_llm = False

    def print_cost_estimate(self):
        """Print estimated cost for running LongMemEval."""
        provider = self.config.embeddings.provider_type
        model = self.config.embeddings.model_name

        if provider == "sentence_transformers":
            print(f"\n💰 Cost: FREE (using local SentenceTransformers)")
            print(f"   Model: {model}")
            print(f"   Device: {self.config.embeddings.device or 'CPU'}")
            print(f"\n   First run will download the model (~600MB)")
        else:
            print(f"\n💰 Cost: ~$0.50 (using OpenAI embeddings)")
            print(f"   Requires: OPENAI_API_KEY environment variable")

        if self.use_llm:
            print(f"\n   LLM Generation: ~$5-10 (GPT-4o for 500 questions)")
            print(f"   Requires: OPENAI_API_KEY environment variable")

    async def evaluate_single(
        self,
        sample_index: int,
        ctx: Optional[Contexo] = None,
        save_progress: bool = True,
    ) -> dict[str, Any]:
        """Evaluate a single question and save progress.

        Args:
            sample_index: Index of the question to evaluate
            ctx: Reusable Contexo instance (if None, creates new one)
            save_progress: Whether to save results incrementally

        Returns:
            Result dictionary with question_id, hypothesis, and metadata
        """
        # Check if already evaluated
        if Path(self.results_path).exists():
            with open(self.results_path) as f:
                existing_ids = set()
                for line in f:
                    if line.strip():
                        result = json.loads(line)
                        existing_ids.add(result.get("question_id"))

            question_id = self.data[sample_index]["question_id"]
            if question_id in existing_ids:
                return {"question_id": question_id, "skipped": True}

        sample = self.data[sample_index]
        question_id = sample["question_id"]
        question_type = sample["question_type"]
        question = sample["question"]
        answer = sample["answer"]

        # Create conversation ID for this question
        conversation_id = f"longmemeval_{question_id}"

        # Create or reuse Contexo instance
        close_on_exit = False
        if ctx is None:
            ctx = Contexo(config=self.config)
            await ctx.initialize()
            close_on_exit = True
        ctx.set_conversation_id(conversation_id)

        existing_count = await ctx._persistent._storage.count(collection=conversation_id)

        # Insert sessions if not already done
        haystack_sessions = sample.get("haystack_sessions", [])
        session_ids = sample.get("haystack_session_ids", [])
        session_dates = sample.get("haystack_dates", [])

        if existing_count == 0 and haystack_sessions:
            # Insert all sessions with temporal metadata
            entries_to_add = []
            for session, sess_id, sess_date in zip(haystack_sessions, session_ids, session_dates):
                for turn in session:
                    entry = MemoryEntry(
                        entry_type=EntryType.MESSAGE,
                        content=turn["content"],
                        metadata={
                            "role": turn["role"],
                            "session_id": sess_id,
                            "session_date": sess_date,
                            "has_answer": turn.get("has_answer", False),
                            "question_id": question_id,
                        },
                        conversation_id=conversation_id,
                    )
                    entries_to_add.append(entry)

            # Batch add all entries for this question
            for entry in entries_to_add:
                await ctx._persistent.add(entry)

        # Search for relevant context
        retrieved = await ctx.search_memory(query=question, limit=50)

        # Check if we retrieved any evidence sessions
        answer_session_ids = set(sample.get("answer_session_ids", []))
        retrieved_session_ids = set()
        for entry in retrieved:
            retrieved_session_ids.add(entry.metadata.get("session_id"))

        found_evidence = bool(answer_session_ids & retrieved_session_ids)

        result = {
            "question_id": question_id,
            "question_type": question_type,
            "question": question,
            "answer": answer,
            "retrieved_count": len(retrieved),
            "retrieved_session_ids": list(retrieved_session_ids),
            "answer_session_ids": list(answer_session_ids),
            "found_evidence": found_evidence,
            "evidence_recall": len(answer_session_ids & retrieved_session_ids) / len(answer_session_ids) if answer_session_ids else 0,
        }

        # Generate hypothesis using LLM if enabled
        if self.use_llm and self._llm_client:
            hypothesis = await self._generate_hypothesis(question, retrieved[:20])
            result["hypothesis"] = hypothesis
        else:
            result["hypothesis"] = "[Retrieval only - no LLM generation]"

        if close_on_exit:
            await ctx.close()

        # Save progress incrementally
        if save_progress:
            self._save_result(result)

        return result

    def _save_result(self, result: dict[str, Any]) -> None:
        """Save a single result to the output file.

        Args:
            result: Result dictionary to save
        """
        # Prepare output line (JSONL format for LongMemEval evaluation)
        output_line = {
            "question_id": result["question_id"],
            "hypothesis": result.get("hypothesis", ""),
        }

        # Append to results file
        with open(self.results_path, "a") as f:
            f.write(json.dumps(output_line) + "\n")

    async def _generate_hypothesis(
        self,
        question: str,
        retrieved: list[MemoryEntry],
    ) -> str:
        """Generate answer hypothesis using LLM.

        Args:
            question: The question to answer
            retrieved: Retrieved context entries

        Returns:
            Generated answer
        """
        # Format retrieved context
        context_parts = []
        for entry in retrieved:
            role = entry.metadata.get("role", "unknown")
            content = entry.content[:500]  # Truncate for context limit
            context_parts.append(f"{role}: {content}")

        context = "\n".join(context_parts)

        # Generate answer
        prompt = f"""Based on the following conversation history, answer the question.

Conversation History:
{context}

Question: {question}

Answer:"""

        try:
            response = await self._llm_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer the question based on the conversation history provided."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  LLM generation failed: {e}")
            return "[LLM generation failed]"

    async def evaluate_all(
        self,
        max_questions: int = None,
    ) -> dict[str, Any]:
        """Evaluate all or specific questions.

        Args:
            max_questions: Maximum number of questions to eval (None = all)

        Returns:
            Summary results
        """
        if max_questions is None:
            sample_indices = list(range(len(self.data)))
        else:
            sample_indices = list(range(min(max_questions, len(self.data))))

        print(f"Will evaluate {len(sample_indices)} questions (max: {max_questions or 'all'})")
        print(f"Results will be saved to: {self.results_path}")

        # Create a single Contexo instance to reuse across all questions
        ctx = Contexo(config=self.config)
        await ctx.initialize()

        all_results = []
        evidence_found = 0
        total_with_evidence = 0
        cumulative_recall = []
        question_type_stats = {}

        for i in sample_indices:
            sample = self.data[i]
            qtype = sample['question_type']
            qid = sample['question_id']

            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(sample_indices)}] {qid[:12]}... ({qtype})")
            print(f"Q: {sample['question'][:60]}...")

            result = await self.evaluate_single(i, ctx=ctx)

            if not result.get("skipped"):
                all_results.append(result)

                # Track by question type
                if qtype not in question_type_stats:
                    question_type_stats[qtype] = {"total": 0, "found": 0}
                question_type_stats[qtype]["total"] += 1
                if result.get("found_evidence"):
                    question_type_stats[qtype]["found"] += 1
                    evidence_found += 1
                if result.get("answer_session_ids"):
                    total_with_evidence += 1

                cumulative_recall.append(result.get("evidence_recall", 0))

                print(f"✓ Found evidence: {result.get('found_evidence')}, Recall: {result.get('evidence_recall', 0):.1%}")
            else:
                print(f"⊘ Skipped (already evaluated)")

            # Show progress every 5 questions or at end
            if (i + 1) % 5 == 0 or i == len(sample_indices) - 1:
                avg_recall = sum(cumulative_recall[-5:]) / min(5, len(cumulative_recall))
                print(f"\n=== Progress: {i+1}/{len(sample_indices)} ===")
                print(f"Evidence found: {evidence_found}/{total_with_evidence}")
                print(f"Cumulative recall: {avg_recall:.1%}")

                # By type
                for qt, stats in question_type_stats.items():
                    recall = stats['found'] / stats['total'] if stats['total'] > 0 else 0
                    print(f"  {qt}: {stats['found']}/{stats['total']} ({recall:.1%})")

        await ctx.close()

        # Calculate summary
        summary = {
            "total_evaluated": len(all_results),
            "evidence_found_count": evidence_found,
            "evidence_recall": evidence_found / total_with_evidence if total_with_evidence > 0 else 0,
            "cumulative_recall": cumulative_recall[-1] if cumulative_recall else 0,
            "by_question_type": question_type_stats,
            "per_question_results": all_results,
        }

        # Save summary
        summary_path = self.results_path.replace(".jsonl", "_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Save incremental progress
        progress_path = self.results_path.replace(".jsonl", "_progress.json")
        with open(progress_path, "w") as f:
            json.dump({
                "max_questions": max_questions,
                "evaluated_indices": sample_indices,
                "total_evaluated": len(all_results),
                "evidence_found_count": evidence_found,
                "evidence_recall": summary["evidence_recall"],
                "by_question_type": question_type_stats,
            }, f, indent=2)

        return summary

    def print_report(self, results: dict[str, Any]) -> None:
        """Print evaluation report.

        Args:
            results: Summary results from evaluate_all
        """
        print("\n" + "=" * 60)
        print("LONGMEMEVAL BENCHMARK RESULTS")
        print("=" * 60)
        print(f"\nTotal Evaluated: {results['total_evaluated']}")
        print(f"Evidence Found: {results['evidence_found_count']}")
        print(f"Evidence Recall: {results['evidence_recall']:.2%}")

        # By question type
        type_stats = {}
        for result in results.get("per_question_results", []):
            qtype = result.get("question_type", "unknown")
            if qtype not in type_stats:
                type_stats[qtype] = {"total": 0, "found": 0}
            type_stats[qtype]["total"] += 1
            if result.get("found_evidence"):
                type_stats[qtype]["found"] += 1

        print("\nBy Question Type:")
        print("-" * 60)
        for qtype, stats in sorted(type_stats.items()):
            recall = stats["found"] / stats["total"] if stats["total"] > 0 else 0
            print(f"{qtype}: {recall:.2%} ({stats['found']}/{stats['total']})")


async def main():
    """Run LongMemEval evaluation."""

    # Check for help
    import sys
    if "--help" in sys.argv or "-h" in sys.argv:
        print("""
LongMemEval Evaluation for Contexo
==================================

Usage:
  python longmemeval_eval.py [options]

Options:
  --max-questions=N    Evaluate N questions (default: 120)
  --all                Evaluate all 500 questions
  --llm                Use LLM to generate answers (requires OPENAI_API_KEY)
  --help, -h           Show this help message

Examples:
  python longmemeval_eval.py              # Evaluate 120 questions
  python longmemeval_eval.py --max-questions=50
  python longmemeval_eval.py --all        # Evaluate all 500 questions
  python longmemeval_eval.py --max-questions=20 --llm
""")
        return

    script_dir = Path(__file__).parent
    data_path = script_dir / "data" / "longmemeval_s_cleaned.json"

    if not data_path.exists():
        print(f"LongMemEval data not found at {data_path}")
        print("\nDownload it first:")
        print("  cd benchmarks/data")
        print("  wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json")
        print("\nOr download all files:")
        print("  wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json")
        print("  wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_m_cleaned.json")
        return

    # Check command-line arguments (sys already imported above)
    use_llm = "--llm" in sys.argv

    # Parse max_questions argument (default: 120)
    max_questions = 120
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--max-questions="):
            max_questions = int(arg.split("=")[1])
        elif arg == "--max-questions" and i + 1 < len(sys.argv):
            max_questions = int(sys.argv[i + 1])
        elif arg == "--all":
            max_questions = None

    runner = LongMemEvalRunner(str(data_path), use_llm=use_llm)
    runner.print_cost_estimate()

    # Check for existing results
    if Path(runner.results_path).exists():
        print(f"\nFound existing results at {runner.results_path}")
        print("Press Enter to continue (will skip completed questions), or '0' to restart")
        response = input("Choice (Enter=continue, 0=restart): ").strip()

        if response == "0":
            Path(runner.results_path).unlink(missing_ok=True)

    # Run evaluation
    print(f"\nStarting LongMemEval evaluation!")
    print(f"Total questions in dataset: {len(runner.data)}")
    print(f"Will evaluate: {max_questions if max_questions else 'all'} questions")

    results = await runner.evaluate_all(max_questions=max_questions)
    runner.print_report(results)

    print(f"\nResults saved to: {runner.results_path}")
    print(f"Summary saved to: {runner.results_path.replace('.jsonl', '_summary.json')}")

    if not use_llm:
        print("\nTo evaluate QA correctness with GPT-4o:")
        print("  export OPENAI_API_KEY=your_key")
        print(f"  cd {Path(__file__).parent.name}/../longmemeval  # or the LongMemEval repo")
        print(f"  python3 src/evaluation/evaluate_qa.py gpt-4o {runner.results_path} data/longmemeval_oracle.json")


if __name__ == "__main__":
    asyncio.run(main())
