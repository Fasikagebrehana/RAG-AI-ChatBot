# import requests
# from rouge_score import rouge_scorer
# from nltk.translate.bleu_score import sentence_bleu
# import time

# # Define test cases with expected answers
# test_cases = [
#     {
#         "query": "What are the penalties of rape?",
#         "expected": "Rigorous imprisonment from three years to fifteen years."
#     },
#     {
#         "query": "What is the penalty for theft?",
#         "expected": "No penalty or relevant information specified."
#     },
#     {
#         "query": "What does the Family Code say about marriage agreements?",
#         "expected": "No penalty or relevant information specified."
#     },
#     {
#         "query": "What’s the weather today?",
#         "expected": "No relevant information found."
#     },
# ]

# # Initialize scorers
# rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# def compute_bleu(reference, candidate):
#     reference = reference.split()
#     candidate = candidate.split()
#     return sentence_bleu([reference], candidate, weights=(0.5, 0.5))

# # Gradio endpoint
# url = "http://127.0.0.1:7860/run"

# for case in test_cases:
#     query = case["query"]
#     expected = case["expected"]
    
#     start = time.time()
#     response = requests.post(url, json={"data": [query]})
#     latency = time.time() - start
    
#     if response.status_code == 200:
#         try:
#             data = response.json()
#             # Extract answer from Gradio's response (first line of output)
#             actual = data["data"][0].split("\n")[0].strip()
#             if not actual:  # Handle empty or malformed response
#                 actual = "No response generated"
#         except (KeyError, IndexError, ValueError):
#             actual = "Error parsing response"
        
#         bleu_score = compute_bleu(expected, actual)
#         rouge_score = rouge_scorer.score(expected, actual)['rougeL'].fmeasure
        
#         print(f"Query: {query}")
#         print(f"Expected: {expected}")
#         print(f"Actual: {actual}")
#         print(f"BLEU Score: {bleu_score:.2f}")
#         print(f"ROUGE-L F1 Score: {rouge_score:.2f}")
#         print(f"Latency: {latency:.2f} seconds")
#         print(f"HTTP Status: {response.status_code}")
#         print("-" * 50)
#     else:
#         print(f"Query failed for '{query}': HTTP {response.status_code}")