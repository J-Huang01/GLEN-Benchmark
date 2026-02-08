import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm


GRAPH_PATH = "./graph/hetero_graph.pt"
SAVE_PATH = "./graphs/hetero_graph_embed.pt"
USE_BGE = True   
USE_PCA = False     
PCA_DIM = 100

graph = torch.load(GRAPH_PATH, weights_only=False)


if USE_BGE:
    model = SentenceTransformer("BAAI/bge-base-en")
else:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

model.to("cuda")
model.eval()

def encode_sentences(sentences, batch_size=16):
    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        convert_to_tensor=True, 
        show_progress_bar=True
    )
    return embeddings


df_demo = pd.read_csv("./data/user_info_data.csv")
df_demo['SEQN'] = df_demo['SEQN'].astype(int)
df_demo = df_demo.set_index('SEQN')

df_demo['user_prompt'] = df_demo.apply(lambda r: f"User {r.name}, age group {r['age_group']}, gender {r['gender']}, race{r['race']}, education{r['education']}, BMI{r['BMI']}", axis=1)

print(graph['user'].node_id[:20])

node_ids = graph['user']['node_id']
ordered_prompts = df_demo.loc[node_ids]['user_prompt']
graph['user'].prompt = ordered_prompts.tolist()

ordered_features = df_demo.drop(['user_prompt', 'label'], axis=1,errors='ignore')
ordered_features = ordered_features.fillna(0)
feature_tensor = torch.tensor(ordered_features.loc[node_ids].values, dtype=torch.float32)

graph['user'].x = feature_tensor

df_food = pd.read_csv("./data/final_food_list.csv")
df_food['food_id'] = df_food['food_id'].astype(str)
df_food = df_food.set_index("food_id")

# gen prompt
df_food['food_prompt'] = df_food.apply(lambda r: f"Food {r.name}, {r['food_desc']}, category {r['WWEIA_desc']}", axis=1)

food_node_ids = graph['food']['node_id']
ordered_food_prompts = df_food.loc[food_node_ids]['food_prompt']
graph['food'].prompt = ordered_food_prompts.tolist()

# nutrition 
numeric_cols = df_food.drop(['food_prompt', 'food_desc', 'WWEIA_desc'], axis=1, errors='ignore').select_dtypes(include=[np.number]).columns
food_nutrition = df_food[numeric_cols]
food_nutrition = food_nutrition.fillna(0)
scaler = StandardScaler()
food_nutrition = pd.DataFrame(scaler.fit_transform(food_nutrition), index=df_food.index, columns=numeric_cols)

# align
food_feature_tensor = torch.tensor(food_nutrition.loc[food_node_ids].values, dtype=torch.float32)
food_feature_tensor = food_feature_tensor.to("cuda")
# embedding
food_embeddings = encode_sentences(graph['food'].prompt)
if USE_PCA:
    reduced = pca.fit_transform(food_embeddings.numpy())
    food_embeddings = torch.tensor(reduced, dtype=torch.float32)

graph['food'].x = torch.cat([food_feature_tensor, food_embeddings], dim=1)

def process_node_with_text(graph, node_type, df, id_col, text_col, prompt_func):
    df[id_col] = df[id_col].astype(str)
    df = df.set_index(id_col)
    df[f'{node_type}_prompt'] = df.apply(prompt_func, axis=1)

    node_ids = [str(x) for x in graph[node_type]['node_id']]
    df.index = df.index.astype(str)

    ordered_prompts = df.loc[node_ids][f'{node_type}_prompt']
    graph[node_type].prompt = ordered_prompts.tolist()

    embeddings = encode_sentences(graph[node_type].prompt)
    if USE_PCA:
        reduced = pca.fit_transform(embeddings.cpu().numpy())
        embeddings = torch.tensor(reduced, dtype=torch.float32).to("cuda")

    graph[node_type].x = embeddings

  
process_node_with_text(
    graph,
    node_type="ingredient",
    df=pd.read_csv("./data/fndds_with_price.csv"),
    id_col="ingredient_id",
    text_col="ingredient_desc",
    prompt_func=lambda r: f"Ingredient {r.name}, {r['ingredient_desc']}"
)

process_node_with_text(
    graph,
    node_type="category",
    df=pd.read_csv("./data/fndds_with_price.csv"),
    id_col="WWEIA_id",
    text_col="WWEIA_desc",
    prompt_func=lambda r: f"Category {r.name}, {r['WWEIA_desc']}"
)

df_habit = pd.read_csv("./data/trimmedHabit.csv")
df_habit = df_habit.drop_duplicates(subset=["habitID"], keep="first")

process_node_with_text(
    graph,
    node_type="habit",
    df=df_habit,
    id_col="habitID",
    text_col="habitDesc",
    prompt_func=lambda r: f"Habit {r.name}, {r['habitDesc']}"
)

def process_text_nodes(graph, node_type, prefix):
    node_ids = graph[node_type]['node_id']
    prompts = [f"{prefix}: {name}" for name in node_ids]
    graph[node_type].prompt = prompts

    embeddings = encode_sentences(prompts)
    if USE_PCA:
        reduced = pca.fit_transform(embeddings.cpu().numpy())
        embeddings = torch.tensor(reduced, dtype=torch.float32).to("cuda")

    graph[node_type].x = embeddings


for node_type, prefix in [
    ("health_condition", "Health condition"),
    ("nutrition_tag", "Nutrition tag"),
    ("poverty_condition", "Poverty condition"),
    ("price_tag", "Price tag")
]:
    node_ids = [str(x) for x in graph[node_type].node_id]
    prompts = [f"{prefix}: {name}" for name in node_ids]
    graph[node_type].prompt = prompts

    embeddings = encode_sentences(prompts)
    if USE_PCA:
        reduced = pca.fit_transform(embeddings.cpu().numpy())
        embeddings = torch.tensor(reduced, dtype=torch.float32)

    graph[node_type].x = embeddings.cpu()  


for ntype in graph.node_types:
    if 'x' in graph[ntype]:
        graph[ntype].x = torch.nan_to_num(
            graph[ntype].x, nan=0.0, posinf=0.0, neginf=0.0
        )
        
torch.save(graph, SAVE_PATH)

for ntype in graph.node_types:
    print(f"{ntype}: {graph[ntype].num_nodes}, x shape: {tuple(graph[ntype].x.shape) if 'x' in graph[ntype] else None}")

for etype in graph.edge_types:
    print(f"{etype}: {graph[etype].edge_index.size(1)} edges")

print(graph)
