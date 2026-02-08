import os
from dotenv import load_dotenv
load_dotenv()
import argparse
from dataset import Dataset
from model import Retriever, Augmenter, Generator, RetrievalEvaluator
from evaluate import Evaluator

import warnings
import logging
import absl.logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)

def main():
    

    # Define note and method prompts
    note_prompts = {
        "easy": "Important Note: Your output will strictly be Yes or No with no other words.",
        "medium": "Important Note: Your output must be strictly formatted as a comma-separated list of nutrients prefixed with “high” or “low”, based solely on the provided options: carb, protein, sugar, sodium, cholesterol, saturated_fat, calorie. \
            For example, a valid output would be: high_carb, low_protein, high_sugar. No extra words or deviations are allowed.",
        "hard": "Important Note: Your output must consist of “Yes” or “No”, followed by a list of nutrients addressed with “high” or “low,” selected from the following options: carb, protein, sugar, sodium, cholesterol, saturated fat, and calorie. \
            For example, a valid output would be: Yes, because the food is high in carb, low in protein, high in sugar. Ensure the output adheres to this format without any additional words or deviations.",
    }

    method_prompts = {
        "plain": "Below are the extra information you use to answer the question, note that you should not use your general knowledge and the answer is among this information.",
        "KAPING": "Below are the extra information you use to answer the question, note that you should not use your general knowledge and the answer is among this information.",
        "ToG": "Below are the extra information you use to answer the question, note that you should not use your general knowledge and the answer is among this information.", 
        "CoT_Zero": "Below are the extra information you use to answer the question, note that you should not use your general knowledge and the answer is among this information. Let's think step by step to determine the healthiness of the food, by extracting the nutritional properties of the food from the given graph, then comparing them to the nutrition requirements of the health status, dietary need and habits of the user. Do not be too strict with your criteria, since not all nutritional tags are important in determining the food's healthiness.",
        "CoT_BaG": "Below are the extra information you use to answer the question, note that you should not use your general knowledge and the answer is among this information. You will be given the textual description of a directed graph.",
        "ToT": "Below is extra information retrieved using Tree of Thoughts reasoning (branching search with evaluation). Use it to answer the question.",
        "GoT": "Below is extra information retrieved using Graph of Thoughts reasoning (multiple thoughts + aggregation). Use it to answer the question.",
        "G_Retriever": "Below is a subgraph retrieved using Prize-Collecting Steiner Tree optimization, balancing relevance and compactness. Use it to answer the question.",
        "KAR": "Below is knowledge-aware retrieved information. Entities were parsed, relations were filtered by document-aware similarity, and the query may be expanded. Use it to answer the question.",
    }

    data = Dataset(args.file_path)
    for question_level in args.question_levels:
        for task_level in args.task_levels:
            for method in args.methods:
                print(f"\nProcessing: Question Level={question_level}, Task Level={task_level}, Method={method}, model={args.model_name}")
                # Process dataset
                questions, answers, graphs = data.process(question_level=question_level, task_level=task_level, sample=args.is_sample, n=args.n)
                # Retrieve subgraphs
                retriever = Retriever(
                    graphs,
                    model_name=args.model_name,
                    embed_backend=args.embed_backend,
                    embed_model=args.embed_model,
                    embed_device=args.embed_device,
                    enable_embedding_cache=args.enable_embedding_cache,
                )

                retrieved_graphs = retriever.retrieve(
                    method=method,
                    api_key=args.api_key,
                    questions=questions,

                    # ToT / GoT knobs
                    tot_depth=args.tot_depth,
                    tot_branching=args.tot_branching,
                    tot_use_llm_eval=args.tot_use_llm_eval,
                    tot_llm_weight=args.tot_llm_weight,

                    got_iterations=args.got_iterations,
                    got_thoughts=args.got_thoughts,
                    got_use_llm_distill=args.got_use_llm_distill,

                    # G-Retriever knobs
                    g_retriever_top_k=args.g_retriever_top_k,
                    g_retriever_beta=args.g_retriever_beta,
                    g_node_cand_k=args.g_node_cand_k,
                    g_edge_cand_k=args.g_edge_cand_k,
                    g_prize_k_node=args.g_prize_k_node,
                    g_prize_k_edge=args.g_prize_k_edge,
                    g_base_edge_cost=args.g_base_edge_cost,
                    g_important_rel_cost=args.g_important_rel_cost,
                    pcst_pruning=args.pcst_pruning,

                    # KAR knobs
                    kar_top_k=args.kar_top_k,
                    kar_use_llm=args.kar_use_llm,
                    kar_use_llm_expand=args.kar_use_llm_expand,
                    kar_hops=args.kar_hops,
                    kar_seed_topm=args.kar_seed_topm,
                    kar_rel_topk=args.kar_rel_topk,
                )
                # Evaluate retrieval
                retrieval_evaluator = RetrievalEvaluator(graphs)
                retrieval_evaluation_results = retrieval_evaluator.evaluate(retrieved_graphs)
                
                print(f"Retrieval evaluation results for Question Level={question_level}, Task Level={task_level}, Method={method}:")
                for metric, value in retrieval_evaluation_results.items():
                    print(f"{metric}: {value}")
                    
                augmenter = Augmenter(method)
                textualized_graphs = augmenter.augment(retrieved_graphs)
                note_prompt = note_prompts.get(task_level)
                method_prompt = method_prompts.get(method)

                generator = Generator(
                    api_key=args.api_key,
                    model_name=args.model_name,
                    note_prompt=note_prompt,
                    method_prompt=method_prompt,
                    sleeptime=1,              
                    save_dir="results"        
                )

                save_name = f"{method}_{task_level}_{question_level}_{args.model_name.replace('/', '_')}.csv"

                predictions = generator.generate_predictions(
                    questions,
                    textualized_graphs,
                    ground_truths=answers, 
                    save_name=save_name
                )

                evaluator = Evaluator()
                final_output_evaluation_results = evaluator.evaluate(task_level, predictions, answers)

                print(f"Final output evaluation results for Question Level={question_level}, Task Level={task_level}, Method={method}:")
                for metric, value in final_output_evaluation_results.items():
                    print(f"{metric}: {value}")
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-level NutriGraphQA benchmark evaluation.")
    parser.add_argument("--file_path", type=str, default="./nutritional_QA/qa_bench_test.csv", help="Path to the dataset file.")
    parser.add_argument("--api_key", type=str, 
                        default=os.getenv('API_KEY'), 
                        help="API key for the model.")
    parser.add_argument("--model_name", type=str, 
                        default="openai/gpt-4o-mini",
                        help="Model name for generation.")
    parser.add_argument("--is_sample", type=bool, default=True, help="Whether to sample data or use the full dataset.")
    parser.add_argument("--n", type=int, default=2000, help="Number of rows to sample if sampling is enabled.")
    
    parser.add_argument("--task_levels", nargs="+", default=['hard'], help="List of task levels to evaluate.")
    parser.add_argument("--question_levels", nargs="+", default=['hard'], help="List of question levels to evaluate.")
    parser.add_argument("--methods", nargs="+", default=["ToG"], help="List of methods to use for retrieval.")

    parser.add_argument("--embed_backend", type=str, default="sentence_transformers", choices=["sentence_transformers", "tfidf"])
    parser.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--embed_device", type=str, default=None)
    parser.add_argument("--enable_embedding_cache", action="store_true")

    # ToT
    parser.add_argument("--tot_depth", type=int, default=3)
    parser.add_argument("--tot_branching", type=int, default=5)
    parser.add_argument("--tot_use_llm_eval", action="store_true", help="If set, ToT makes 1 LLM call per depth.")
    parser.add_argument("--tot_llm_weight", type=float, default=1.0)

    # GoT
    parser.add_argument("--got_iterations", type=int, default=3)
    parser.add_argument("--got_thoughts", type=int, default=5)
    parser.add_argument("--got_use_llm_distill", action="store_true", help="If set, GoT makes 1 LLM call to distill thoughts.")

    # G-Retriever
    parser.add_argument("--g_retriever_top_k", type=int, default=15)
    parser.add_argument("--g_retriever_beta", type=float, default=1.0)
    parser.add_argument("--g_node_cand_k", type=int, default=30)
    parser.add_argument("--g_edge_cand_k", type=int, default=30)
    parser.add_argument("--g_prize_k_node", type=int, default=20)
    parser.add_argument("--g_prize_k_edge", type=int, default=20)
    parser.add_argument("--g_base_edge_cost", type=float, default=1.0)
    parser.add_argument("--g_important_rel_cost", type=float, default=0.5)
    parser.add_argument("--pcst_pruning", type=str, default="gw", choices=["none", "simple", "gw", "strong"])

    # KAR
    parser.add_argument("--kar_top_k", type=int, default=15)
    parser.add_argument("--kar_use_llm", action="store_true", help="If set, KAR uses one LLM call for entity parsing.")
    parser.add_argument("--kar_use_llm_expand", action="store_true", help="If set, KAR uses one LLM call for query expansion.")
    parser.add_argument("--kar_hops", type=int, default=2)
    parser.add_argument("--kar_seed_topm", type=int, default=2)
    parser.add_argument("--kar_rel_topk", type=int, default=30)

    
    args = parser.parse_args()

    main()
