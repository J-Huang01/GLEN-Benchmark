import os
import time
import logging
from tqdm import tqdm
import networkx as nx
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from typing import List, Dict, Optional

from utils import (
    find_relations, find_entities, convert_to_sg, generate_paragraph_cot_bag
)

import warnings
warnings.filterwarnings("ignore")

class LlamaLocalClient:
    def __init__(
        self,
        model_path: str,
        token: Optional[str] = None,
        device: str = "cuda",
        use_quantization: bool = True
    ):
        self.model_path = model_path
        self.token = token
        self.device = device
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        print(f"⏳ Loading Llama model from {model_path}...")

        try:
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU Memory: {total_mem:.1f} GB")
        except:
            total_mem = 0
            print("No GPU detected, will try CPU mode")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            token=token,
            local_files_only=True
        )
        
        if use_quantization and total_mem > 0 and total_mem < 80:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quant_config,
                device_map="auto",
                token=token,
                local_files_only=True
            )
        else:
            print("Loading full precision model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                token=token,
                local_files_only=True
            )
        
        self.model.eval()
        print("Llama model loaded successfully!")
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        prompt = "<|begin_of_text|>"
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 512,
        **kwargs
    ) -> str:
        prompt = self._format_messages(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=kwargs.get("top_p", 0.9),
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        if "assistant" in text:
            text = text.split("assistant")[-1].strip()
        
        return text


class LlamaOpenAIAdapter:
    def __init__(self, model_path: str, token: Optional[str] = None, **kwargs):
        self.llama_client = LlamaLocalClient(model_path, token, **kwargs)
        self.model_path = model_path
    
    class ChatCompletions:
        def __init__(self, llama_client):
            self.llama_client = llama_client
        
        def create(self, model: str, messages: List[Dict], **kwargs):
            response_text = self.llama_client.generate(messages, **kwargs)
    
            class Message:
                def __init__(self, content):
                    self.content = content
            
            class Choice:
                def __init__(self, content):
                    self.message = Message(content)
            
            class Response:
                def __init__(self, content):
                    self.choices = [Choice(content)]
            
            return Response(response_text)
    
    @property
    def chat(self):
        class Chat:
            def __init__(self, llama_client):
                self.completions = LlamaOpenAIAdapter.ChatCompletions(llama_client)
        
        return Chat(self.llama_client)

def convert_to_txt(path):
    full_path = path[:1]
    for i in range(1, len(path) - 1):
        if i % 2 == 1:
            full_path.append(path[i])
        else:
            full_path.extend([path[i], ',', path[i]])
    full_path.append(path[-1])
    return ' '.join(full_path)


def prune_relations(client, path_list, question, model_name, width):
    import re
    
    if len(path_list) <= width:
        return path_list
    
    reasoning_path_list = [f'{i + 1}. ' + convert_to_txt(path) + '.\n' for i, path in enumerate(path_list)]
    
    messages = [
        {
            'role': 'system',
            'content': f'Identify the top-{width} reasoning paths that are most likely to lead to the answer for the query. \
                        Respond with the indices of the reasoning paths, starting from 1, and separate them with commas (e.g., 1,2,5). Include nothing else in your response.'
        },
        {
            'role': 'user', 
            'content': f'The query is {question}, and the reasoning paths are: \n{reasoning_path_list}. Your selected top-{width} reasoning paths are:' 
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
        )
        answer = response.choices[0].message.content
    except Exception as e:
        print(f"Error in prune_relations: {e}")
        return path_list[:width]

    indices = re.findall(r'\d+', answer)
    indices = [int(index) - 1 for index in indices]
    if len(indices) > width:
        indices = indices[:width]
    
    indices = [index for index in indices if index < len(path_list)]
    
    if len(indices) == 0:
        return path_list[:width]
    
    path_list = [path_list[index] for index in indices]
    return path_list


def prune_entities(client, path_list, question, model_name, width):
    import re
    
    if len(path_list) <= width:
        return path_list
    
    reasoning_path_list = [f'{i + 1}. ' + convert_to_txt(path) + '.\n' for i, path in enumerate(path_list)]

    messages = [
        {
            'role': 'system',
            'content': f'Identify the top-{width} reasoning paths that are most likely to lead to the answer for the query. \
                        Respond with the indices of the reasoning paths, starting from 1, and separate them with commas (e.g., 1,2,5). Include nothing else in your response.'
        },
        {
            'role': 'user', 
            'content': f'The query is {question}, and the reasoning paths are: \n{reasoning_path_list}. Your selected top-{width} reasoning paths are:' 
        }
    ]
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
        )
        answer = response.choices[0].message.content
    except Exception as e:
        print(f"Error in prune_entities: {e}")
        return path_list[:width]

    indices = re.findall(r'\d+', answer)
    indices = [int(index) - 1 for index in indices]
    if len(indices) > width:
        indices = indices[:width]
    
    indices = [index for index in indices if index < len(path_list)]
        
    if len(indices) == 0:
        return path_list[:width]
    
    path_list = [path_list[index] for index in indices]
    return path_list

class Retriever:
    def __init__(self, graphs, model_name='', api_key=None, llama_model_path=None):
        self.graphs = graphs
        self.model_name = model_name
        self.api_key = api_key
        self.llama_model_path = llama_model_path
        self.client = None

    def _get_client(self):
        if self.client is None:
            if 'gpt' in self.model_name.lower():
                self.client = OpenAI(api_key=self.api_key)
            elif 'llama' in self.model_name.lower():
                if self.llama_model_path is None:
                    raise ValueError("llama_model_path must be provided for Llama models in ToG retrieval")
                self.client = LlamaOpenAIAdapter(
                    model_path=self.llama_model_path,
                    token=self.api_key or os.getenv("HF_TOKEN")
                )
            else:
                raise ValueError(f"Unsupported model for ToG: {self.model_name}")
        return self.client

    def plain_retriever(self, graph):
        return graph

    def KAPING_retriever(self, graph):
        subgraph = nx.DiGraph()
        for source in [0, 1]:
            if source in graph:
                subgraph.add_node(source, **graph.nodes[source])
                for neighbor in graph.neighbors(source):
                    subgraph.add_node(neighbor, **graph.nodes[neighbor])
                    subgraph.add_edge(source, neighbor, **graph[source][neighbor])
        return subgraph

    def tog_retriever(self, graph, question):
        depth, width = 2, 3
        
        client = self._get_client()

        reasoning_path_list = [[graph.nodes[0]['attr']], [graph.nodes[1]['attr']]]
        
        for i in range(depth):
            candidate_reasoning_path_list = []
            for path in reasoning_path_list:
                new_paths = find_relations(graph, path)
                candidate_reasoning_path_list.extend(new_paths or [path])

            if i > 0:
                reasoning_path_list = prune_relations(
                    client, 
                    candidate_reasoning_path_list, 
                    question, 
                    self.model_name, 
                    width
                )
            else:
                reasoning_path_list = candidate_reasoning_path_list

            candidate_reasoning_path_list = []
            for path in reasoning_path_list:
                new_paths = find_entities(graph, path)
                candidate_reasoning_path_list.extend(new_paths or [path])

            if i > 0:
                reasoning_path_list = prune_entities(
                    client, 
                    candidate_reasoning_path_list, 
                    question, 
                    self.model_name, 
                    width
                )
            else:
                reasoning_path_list = candidate_reasoning_path_list

            if i == depth - 1:
                return convert_to_sg(graph, reasoning_path_list)

    def retrieve(self, method="plain", questions=None):
        retrieved_graphs = []
        
        for i, graph in tqdm(enumerate(self.graphs), desc=f"Retrieving with {method}", total=len(self.graphs)):
            if method in ["plain", "CoT_Zero", "CoT_BaG"]:
                retrieved_graphs.append(self.plain_retriever(graph))
            elif method == "KAPING":
                retrieved_graphs.append(self.KAPING_retriever(graph))
            elif method == "ToG":
                if questions is None or i >= len(questions):
                    raise ValueError("ToG method requires questions parameter")
                retrieved_graphs.append(self.tog_retriever(graph, questions[i]))
            else:
                raise ValueError(f"Unknown retrieval method: {method}")
        
        return retrieved_graphs


class RetrievalEvaluator:
    def __init__(self, graphs):
        self.graphs = graphs

    def evaluate(self, retrieved_graphs):
        results = {"Precision": 0, "Recall": 0, "F1 Score": 0}
        for r_g, g in zip(retrieved_graphs, self.graphs):
            opt_nodes = set()
            for path in nx.all_simple_paths(g.to_undirected(), 0, 1):
                opt_nodes.update(path)
            opt_names = [g.nodes[i]['attr'] for i in opt_nodes]
            ret_names = [r_g.nodes[i]['attr'] for i in r_g.nodes]
            p = len(set(ret_names) & set(opt_names)) / max(len(ret_names), 1)
            r = len(set(ret_names) & set(opt_names)) / max(len(opt_names), 1)
            f1 = 2 * p * r / (p + r) if (p + r) else 0
            results["Precision"] += p
            results["Recall"] += r
            results["F1 Score"] += f1
        n = len(self.graphs)
        for k in results:
            results[k] = round(results[k] / n, 3)
        return results

class Augmenter:
    def __init__(self, method='plain'):
        self.method = method

    def graph_to_triplets(self, graph):
        triplets = []
        for source, target, data in graph.edges(data=True):
            rel = data.get("relationship", "related_to")
            src = graph.nodes[source].get("attr", str(source))
            tgt = graph.nodes[target].get("attr", str(target))
            triplets.append(f"({src} {rel} {tgt})")
        return ", ".join(triplets)

    def augment(self, graphs, method='to_triplets'):
        texts = []
        for g in graphs:
            t = self.graph_to_triplets(g) if method == 'to_triplets' else ""
            if self.method == "CoT_BaG":
                t = generate_paragraph_cot_bag(t)
            texts.append(t)
        return texts

class Generator:
    def __init__(self, api_key, model_name, note_prompt, method_prompt, 
                 llama_model_path=None, sleeptime=0):
        self.api_key = api_key
        self.model_name = model_name
        self.note_prompt = note_prompt
        self.method_prompt = method_prompt
        self.sleeptime = sleeptime
        self.system_prompt = "Act as a nutritionist. Analyze if a given food is healthy to a user and why."

        logging.basicConfig(level=logging.INFO)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        self.logger = logging.getLogger(__name__)

        if "gpt" in model_name.lower():
            self.client = OpenAI(api_key=self.api_key)
            self.is_llama = False
            print(f"Generator using GPT model: {model_name}")
        elif "llama" in model_name.lower():
            self.is_llama = True
            if llama_model_path is None:
                raise ValueError("llama_model_path must be provided for Llama models")
 
            self.client = LlamaOpenAIAdapter(
                model_path=llama_model_path,
                token=api_key or os.getenv("HF_TOKEN")
            )
            print(f"Generator using Llama model from: {llama_model_path}")
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def generate_prompt(self, question, graph_text):
        return f"{question}\n\n{self.method_prompt}\n\n{graph_text}\n\n{self.note_prompt}"

    def query_api(self, prompt, retries=3, delay=2):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0 if not self.is_llama else 0.2
                )
                return resp.choices[0].message.content
            except Exception as e:
                self.logger.error(f"API Error (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(delay)
        
        return "API Error after multiple retries"

    def generate_predictions(self, questions, graphs):
        if len(questions) != len(graphs):
            raise ValueError("Questions and graphs must match in length.")
        
        preds = []
        for q, g in tqdm(zip(questions, graphs), total=len(questions), desc="Generating Predictions"):
            prompt = self.generate_prompt(q, g)
            preds.append(self.query_api(prompt))
            time.sleep(self.sleeptime)
        
        return preds