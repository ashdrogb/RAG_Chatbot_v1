"""
RAG Evaluation & Diagnostic Tool
Tests retrieval quality, answer quality, and identifies issues
"""

import requests
import json
from typing import List, Dict
from datetime import datetime

# ============================================
# TEST QUESTIONS
# ============================================

# Create test questions with known answers from your PDFs
TEST_QUESTIONS = [
    {
        "question": "What was the fiscal deficit target for 2020-21?",
        "expected_sources": ["Budget_2020_21.pdf"],
        "expected_keywords": ["fiscal deficit", "3.5%", "GDP"],
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
        "expected_keywords": ["education", "allocation", "budget"],
        "category": "specific_fact"
    },
    {
        "question": "Compare tax policies between 2020 and 2021",
        "expected_sources": ["Budget_2020_21.pdf", "Budget_2021_22.pdf"],
        "expected_keywords": ["tax", "policy", "2020", "2021"],
        "category": "comparison"
    },
    {
        "question": "What is the capital of France?",  # Should fail - not in budget docs
        "expected_sources": [],
        "expected_keywords": [],
        "category": "out_of_scope"
    }
]

# ============================================
# EVALUATION FUNCTIONS
# ============================================

def test_retrieval_quality(api_url: str = "http://localhost:8000"):
    """Test if retrieval is finding the right documents"""
    
    print("\n" + "="*70)
    print("🔍 RETRIEVAL QUALITY TEST")
    print("="*70)
    
    results = []
    
    for test in TEST_QUESTIONS:
        print(f"\n❓ Question: {test['question']}")
        print(f"   Expected sources: {test['expected_sources']}")
        
        # Send to API
        response = requests.post(
            f"{api_url}/chat",
            json={"message": test["question"]}
        )
        
        if response.status_code != 200:
            print(f"   ❌ API Error: {response.status_code}")
            continue
        
        data = response.json()
        retrieved_sources = [s["source"] for s in data["sources"]]
        
        print(f"   Retrieved sources: {retrieved_sources}")
        
        # Calculate metrics
        if test['expected_sources']:
            # Precision: How many retrieved docs are in expected list?
            relevant_retrieved = sum(1 for s in retrieved_sources if s in test['expected_sources'])
            precision = relevant_retrieved / len(retrieved_sources) if retrieved_sources else 0
            
            # Recall: Did we get all expected sources?
            recall = relevant_retrieved / len(test['expected_sources']) if test['expected_sources'] else 0
            
            # Hit rate: Did we get at least one relevant doc?
            hit = relevant_retrieved > 0
            
            print(f"   📊 Precision: {precision:.2f} | Recall: {recall:.2f} | Hit: {hit}")
            
            results.append({
                "question": test["question"],
                "precision": precision,
                "recall": recall,
                "hit": hit,
                "category": test["category"]
            })
        else:
            # Out of scope question - should ideally return low-confidence or generic sources
            print(f"   ⚠️  Out-of-scope question")
    
    # Summary
    print("\n" + "="*70)
    print("📊 RETRIEVAL SUMMARY")
    print("="*70)
    
    if results:
        avg_precision = sum(r["precision"] for r in results) / len(results)
        avg_recall = sum(r["recall"] for r in results) / len(results)
        hit_rate = sum(r["hit"] for r in results) / len(results)
        
        print(f"Average Precision: {avg_precision:.2%}")
        print(f"Average Recall: {avg_recall:.2%}")
        print(f"Hit Rate: {hit_rate:.2%}")
        
        # By category
        print("\n📋 By Category:")
        categories = set(r["category"] for r in results)
        for cat in categories:
            cat_results = [r for r in results if r["category"] == cat]
            cat_precision = sum(r["precision"] for r in cat_results) / len(cat_results)
            print(f"  {cat}: Precision = {cat_precision:.2%}")
    
    return results


def test_answer_quality(api_url: str = "http://localhost:8000"):
    """Test if generated answers contain expected information"""
    
    print("\n" + "="*70)
    print("💬 ANSWER QUALITY TEST")
    print("="*70)
    
    results = []
    
    for test in TEST_QUESTIONS:
        if test['category'] == 'out_of_scope':
            continue  # Skip out-of-scope questions
        
        print(f"\n❓ Question: {test['question']}")
        print(f"   Expected keywords: {test['expected_keywords']}")
        
        # Send to API
        response = requests.post(
            f"{api_url}/chat",
            json={"message": test["question"]}
        )
        
        if response.status_code != 200:
            print(f"   ❌ API Error: {response.status_code}")
            continue
        
        data = response.json()
        answer = data["response"].lower()
        
        print(f"   📝 Answer: {data['response'][:100]}...")
        
        # Check keyword coverage
        keywords_found = sum(1 for kw in test['expected_keywords'] if kw.lower() in answer)
        keyword_coverage = keywords_found / len(test['expected_keywords']) if test['expected_keywords'] else 0
        
        # Check answer length (too short = likely hallucination or "I don't know")
        answer_length = len(data['response'].split())
        
        # Check if answer just regurgitates context (bad sign)
        is_regurgitation = len(answer) > 200 and any(
            doc_marker in answer for doc_marker in ['[doc', 'page', 'budget_']
        )
        
        print(f"   📊 Keyword coverage: {keyword_coverage:.2%}")
        print(f"   📏 Answer length: {answer_length} words")
        print(f"   ⚠️  Regurgitation: {is_regurgitation}")
        
        results.append({
            "question": test["question"],
            "keyword_coverage": keyword_coverage,
            "answer_length": answer_length,
            "is_regurgitation": is_regurgitation,
            "answer": data['response']
        })
    
    # Summary
    print("\n" + "="*70)
    print("📊 ANSWER QUALITY SUMMARY")
    print("="*70)
    
    if results:
        avg_coverage = sum(r["keyword_coverage"] for r in results) / len(results)
        avg_length = sum(r["answer_length"] for r in results) / len(results)
        regurgitation_rate = sum(r["is_regurgitation"] for r in results) / len(results)
        
        print(f"Average Keyword Coverage: {avg_coverage:.2%}")
        print(f"Average Answer Length: {avg_length:.1f} words")
        print(f"Regurgitation Rate: {regurgitation_rate:.2%}")
    
    return results


def diagnose_common_issues(api_url: str = "http://localhost:8000"):
    """Diagnose common RAG problems"""
    
    print("\n" + "="*70)
    print("🔧 DIAGNOSTIC TEST - COMMON ISSUES")
    print("="*70)
    
    issues = []
    
    # Test 1: Chunk size issue
    print("\n1️⃣ Testing chunk size...")
    response = requests.post(
        f"{api_url}/chat",
        json={"message": "What was the fiscal deficit?"}
    )
    
    if response.status_code == 200:
        data = response.json()
        answer = data['response']
        
        # Check if answer is just repeating context without synthesis
        if '[Doc' in answer or 'Page' in answer:
            issues.append("❌ ISSUE: Answer contains raw document markers - LLM is copying context")
            print("   ❌ Answer contains document markers")
        
        # Check if answer is incomplete
        if len(answer.split()) < 10:
            issues.append("❌ ISSUE: Very short answers - might need larger chunks or more context")
            print("   ❌ Answer too short (< 10 words)")
    
    # Test 2: Retrieval relevance
    print("\n2️⃣ Testing retrieval relevance...")
    response = requests.post(
        f"{api_url}/chat",
        json={"message": "What is the capital of France?"}
    )
    
    if response.status_code == 200:
        data = response.json()
        answer = data['response'].lower()
        
        # Should ideally say "I don't know" or similar
        if 'paris' in answer or 'france' in answer:
            issues.append("⚠️  WARNING: Answering out-of-scope questions - LLM might be hallucinating")
            print("   ⚠️  Answered out-of-scope question (hallucination risk)")
        elif "don't" not in answer and "cannot" not in answer:
            issues.append("⚠️  WARNING: Not refusing out-of-scope questions properly")
            print("   ⚠️  Should explicitly say 'I don't know' for out-of-scope")
    
    # Test 3: Context window
    print("\n3️⃣ Testing context window...")
    response = requests.post(
        f"{api_url}/chat",
        json={"message": "Summarize all the major points from Budget 2020"}
    )
    
    if response.status_code == 200:
        data = response.json()
        answer = data['response']
        
        # Broad questions need more context
        if len(answer.split()) < 30:
            issues.append("⚠️  WARNING: Short answer for broad question - might need more retrieved docs")
            print("   ⚠️  Broad question got short answer")
    
    # Summary
    print("\n" + "="*70)
    print("🔍 DIAGNOSTIC SUMMARY")
    print("="*70)
    
    if not issues:
        print("✅ No major issues detected!")
    else:
        for issue in issues:
            print(issue)
    
    return issues


def benchmark_speed(api_url: str = "http://localhost:8000", num_queries: int = 5):
    """Measure response times"""
    
    print("\n" + "="*70)
    print("⚡ SPEED BENCHMARK")
    print("="*70)
    
    times = []
    
    test_question = "What were the tax changes in 2020?"
    
    for i in range(num_queries):
        print(f"\nQuery {i+1}/{num_queries}...")
        
        start = datetime.now()
        response = requests.post(
            f"{api_url}/chat",
            json={"message": test_question}
        )
        end = datetime.now()
        
        duration = (end - start).total_seconds()
        times.append(duration)
        
        print(f"   Time: {duration:.2f}s")
    
    # Summary
    print("\n" + "="*70)
    print("📊 SPEED SUMMARY")
    print("="*70)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"Average: {avg_time:.2f}s")
    print(f"Min: {min_time:.2f}s")
    print(f"Max: {max_time:.2f}s")
    
    if avg_time > 10:
        print("\n⚠️  Response time is slow (>10s)")
        print("   Consider: Using a smaller model or GPU")
    elif avg_time > 5:
        print("\n⚠️  Response time is moderate (5-10s)")
        print("   This is normal for CPU inference")
    else:
        print("\n✅ Response time is good (<5s)")
    
    return times


# ============================================
# MAIN RUNNER
# ============================================

def run_full_evaluation(api_url: str = "http://localhost:8000"):
    """Run complete RAG evaluation"""
    
    print("\n" + "="*70)
    print("🚀 STARTING FULL RAG EVALUATION")
    print("="*70)
    print(f"API: {api_url}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if API is running
    try:
        response = requests.get(f"{api_url}/health")
        if response.status_code != 200:
            print("\n❌ ERROR: API health check failed")
            return
    except:
        print("\n❌ ERROR: Cannot connect to API. Is it running?")
        return
    
    print("✅ API is running\n")
    
    # Run all tests
    retrieval_results = test_retrieval_quality(api_url)
    answer_results = test_answer_quality(api_url)
    issues = diagnose_common_issues(api_url)
    times = benchmark_speed(api_url, num_queries=3)
    
    # Final summary
    print("\n" + "="*70)
    print("🎯 FINAL EVALUATION SUMMARY")
    print("="*70)
    
    print("\n📋 Key Findings:")
    
    # Retrieval
    if retrieval_results:
        avg_precision = sum(r["precision"] for r in retrieval_results) / len(retrieval_results)
        if avg_precision < 0.5:
            print("❌ Poor retrieval precision - documents not relevant")
            print("   → Fix: Improve chunking, use better embeddings, or increase k")
        elif avg_precision < 0.8:
            print("⚠️  Moderate retrieval precision")
            print("   → Fix: Fine-tune chunk size and overlap")
        else:
            print("✅ Good retrieval precision")
    
    # Answer quality
    if answer_results:
        avg_coverage = sum(r["keyword_coverage"] for r in answer_results) / len(answer_results)
        if avg_coverage < 0.3:
            print("❌ Low keyword coverage - answers missing key info")
            print("   → Fix: Use better LLM, improve prompt, or retrieve more docs")
        elif avg_coverage < 0.6:
            print("⚠️  Moderate keyword coverage")
            print("   → Fix: Improve prompt engineering")
        else:
            print("✅ Good keyword coverage")
    
    # Speed
    avg_time = sum(times) / len(times)
    if avg_time > 10:
        print(f"❌ Slow responses ({avg_time:.1f}s average)")
        print("   → Fix: Use GPU, smaller model, or optimize retrieval")
    else:
        print(f"✅ Acceptable speed ({avg_time:.1f}s average)")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    run_full_evaluation()
