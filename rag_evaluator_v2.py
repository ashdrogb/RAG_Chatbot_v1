"""
RAG Evaluation & Diagnostic Tool - Improved Version
Adds MRR, nDCG, top-k hit, faithfulness proxy, refusal scoring
"""

import requests
from typing import List, Dict
from datetime import datetime
import math

# ============================================
# CONFIG
# ============================================

API_URL = "http://localhost:8000"
TOP_K_EVAL = 5
MIN_REFUSAL_TOKENS = ["don't know", "not in the provided", "cannot find", "no information"]

# ============================================
# TEST QUESTIONS
# ============================================

TEST_QUESTIONS = [
    {
        "question": "What was the fiscal deficit target for 2020-21?",
        "expected_sources": ["Budget_2020_21.pdf"],
        "expected_keywords": ["fiscal deficit", "3.5%", "gdp"],
        "category": "specific_fact"
    },
    {
        "question": "What were the major tax changes in Budget 2020?",
        "expected_sources": ["Budget_2020_21.pdf"],
        "expected_keywords": ["tax", "income tax", "corporate tax"],
        "category": "broad_topic"
    },
    {
        "question": "How much was allocated for education in 2021?",
        "expected_sources": ["Budget_2021_22.pdf"],
        "expected_keywords": ["education", "allocation"],
        "category": "specific_fact"
    },
    {
        "question": "Compare tax policies between 2020 and 2021",
        "expected_sources": ["Budget_2020_21.pdf", "Budget_2021_22.pdf"],
        "expected_keywords": ["tax", "2020", "2021"],
        "category": "comparison"
    },
    {
        "question": "What is the capital of France?",
        "expected_sources": [],
        "expected_keywords": [],
        "category": "out_of_scope"
    }
]

# ============================================
# HELPERS
# ============================================

def reciprocal_rank(retrieved: List[str], expected: List[str]) -> float:
    for i, doc in enumerate(retrieved):
        if doc in expected:
            return 1 / (i + 1)
    return 0.0

def dcg(retrieved: List[str], expected: List[str]) -> float:
    score = 0.0
    for i, doc in enumerate(retrieved):
        rel = 1 if doc in expected else 0
        score += rel / math.log2(i + 2)
    return score

def idcg(expected_len: int) -> float:
    return sum(1 / math.log2(i + 2) for i in range(expected_len))

def ndcg(retrieved: List[str], expected: List[str]) -> float:
    if not expected:
        return 0.0
    return dcg(retrieved, expected) / idcg(len(expected))

def contains_refusal(answer: str) -> bool:
    answer = answer.lower()
    return any(token in answer for token in MIN_REFUSAL_TOKENS)

# ============================================
# RETRIEVAL EVAL
# ============================================

def test_retrieval_quality():
    print("\n" + "="*70)
    print("🔍 RETRIEVAL QUALITY TEST (Enhanced)")
    print("="*70)

    results = []

    for test in TEST_QUESTIONS:
        print(f"\n❓ {test['question']}")

        response = requests.post(
            f"{API_URL}/chat",
            json={"message": test["question"]}
        )

        if response.status_code != 200:
            print("❌ API error")
            continue

        data = response.json()
        retrieved_sources = list(dict.fromkeys(
            [s["source"] for s in data["sources"]]
        ))[:TOP_K_EVAL]

        expected = test["expected_sources"]

        print("   Retrieved:", retrieved_sources)
        print("   Expected :", expected)

        if not expected:
            print("   ⚠️ Out-of-scope question")
            continue

        relevant = [doc for doc in retrieved_sources if doc in expected]

        precision = len(relevant) / len(retrieved_sources) if retrieved_sources else 0
        recall = len(relevant) / len(expected)
        hit = len(relevant) > 0
        mrr = reciprocal_rank(retrieved_sources, expected)
        ndcg_score = ndcg(retrieved_sources, expected)

        print(f"   📊 P={precision:.2f} R={recall:.2f} Hit={hit} MRR={mrr:.2f} nDCG={ndcg_score:.2f}")

        results.append({
            "precision": precision,
            "recall": recall,
            "hit": hit,
            "mrr": mrr,
            "ndcg": ndcg_score,
            "category": test["category"]
        })

    # Summary
    if results:
        print("\n📊 RETRIEVAL SUMMARY")
        print("="*70)
        for metric in ["precision", "recall", "hit", "mrr", "ndcg"]:
            avg = sum(r[metric] for r in results) / len(results)
            print(f"{metric.upper():<10}: {avg:.2%}")

    return results

# ============================================
# ANSWER EVAL
# ============================================

def test_answer_quality():
    print("\n" + "="*70)
    print("💬 ANSWER QUALITY TEST (Enhanced)")
    print("="*70)

    results = []

    for test in TEST_QUESTIONS:
        print(f"\n❓ {test['question']}")

        response = requests.post(
            f"{API_URL}/chat",
            json={"message": test["question"]}
        )

        if response.status_code != 200:
            continue

        data = response.json()
        answer = data["response"].lower()
        sources = " ".join([s["content"].lower() for s in data["sources"]])

        if test["category"] == "out_of_scope":
            refusal = contains_refusal(answer)
            print(f"   Refusal detected: {refusal}")
            results.append({"refusal": refusal})
            continue

        # Keyword coverage
        keywords_found = sum(1 for kw in test["expected_keywords"] if kw in answer)
        keyword_coverage = keywords_found / len(test["expected_keywords"]) if test["expected_keywords"] else 0

        # Faithfulness proxy: overlap with retrieved context
        overlap = sum(1 for word in answer.split() if word in sources)
        faithfulness = overlap / len(answer.split()) if answer.split() else 0

        # Hallucination proxy: low overlap but long answer
        hallucination_flag = faithfulness < 0.2 and len(answer.split()) > 40

        print(f"   📊 Keyword={keyword_coverage:.2f} Faithfulness={faithfulness:.2f} HallucinationRisk={hallucination_flag}")

        results.append({
            "keyword_coverage": keyword_coverage,
            "faithfulness": faithfulness,
            "hallucination": hallucination_flag
        })

    # Summary
    if results:
        print("\n📊 ANSWER SUMMARY")
        print("="*70)

        if any("keyword_coverage" in r for r in results):
            avg_kw = sum(r.get("keyword_coverage", 0) for r in results) / len(results)
            avg_faith = sum(r.get("faithfulness", 0) for r in results) / len(results)
            halluc_rate = sum(r.get("hallucination", False) for r in results) / len(results)

            print(f"Keyword Coverage : {avg_kw:.2%}")
            print(f"Faithfulness     : {avg_faith:.2%}")
            print(f"HallucinationRate: {halluc_rate:.2%}")

        refusal_rate = sum(r.get("refusal", False) for r in results) / len(results)
        print(f"Refusal Accuracy : {refusal_rate:.2%}")

    return results

# ============================================
# MAIN
# ============================================

def run_full_evaluation():
    print("\n" + "="*70)
    print("🚀 STARTING FULL RAG EVALUATION (Enhanced)")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        r = requests.get(f"{API_URL}/health")
        if r.status_code != 200:
            print("❌ API not healthy")
            return
    except:
        print("❌ Cannot connect to API")
        return

    print("✅ API is running")

    retrieval_results = test_retrieval_quality()
    answer_results = test_answer_quality()

    print("\n🎯 FINAL SUMMARY")
    print("="*70)

    if retrieval_results:
        avg_hit = sum(r["hit"] for r in retrieval_results) / len(retrieval_results)
        if avg_hit < 0.5:
            print("❌ Retriever failing → REINDEX DOCUMENTS")
        else:
            print("✅ Retriever OK")

    if answer_results:
        halluc_rate = sum(r.get("hallucination", False) for r in answer_results) / len(answer_results)
        if halluc_rate > 0.3:
            print("⚠️ High hallucination risk → add relevance threshold")
        else:
            print("✅ Low hallucination risk")


if __name__ == "__main__":
    run_full_evaluation()
