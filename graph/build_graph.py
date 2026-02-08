import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm
import pandas as pd

USER_FILE = "./data/user_tagging_final.csv"
FOOD_FILE = "./data/final_food_list.csv"
FNDDS_FILE = "./data/fndds_with_price.csv"
HABIT_FILE = "./data/trimmedHabit.csv"
FOODUSER_FILE = "./data/food_user.csv"

reference_dict = {
    "anemia": ["high_iron"],
    "ckd": ["low_protein"],
    "dyslipidemia": ["low_saturated_fat"],
    "hyperuricemia": ["low_purine"],
    "sleep_disorder": ["high_vitamin_d"],
    "depression": ["high_folate_acid"],
    "liver_disease": ["low_sodium"],
    'obesity': ['low_calorie'], 
    'hypertension': ['low_sodium'], 
    'diabetes': ['low_sugar', 'low_carb'], 
    'Weight loss/Low calorie diet': ['low_calorie'], 
    'Low fat/Low cholesterol diet': ['low_cholesterol', 'low_saturated_fat'], 
    'Low salt/Low sodium diet': ['low_sodium'], 
    'Sugar free/Low sugar diet': ['low_sugar'], 
    'Diabetic diet': ['low_sugar', 'low_carb'],  
    'Weight gain/Muscle building diet': ['high_calorie', 'high_protein'], 
    'Low carbohydrate diet': ['low_carb'], 
    'High protein diet': ['high_protein'], 
    'Renal/Kidney diet': ['low_protein'],
}

food_primary_nutrition_tags = [
     'low_carb', 'high_carb',
    'low_sugar', 'high_sugar',
    'low_sodium', 'high_sodium',
    'low_calorie', 'high_calorie',
    'low_protein', 'high_protein',
    'low_cholesterol', 'high_cholesterol',
    'low_saturated_fat', 'high_saturated_fat',
]


def build_big_hetero_graph(users, foods, fndds_ingredients, habits, fooduser,
                           reference_dict, food_primary_nutrition_tags):
    data = HeteroData()
    user_labels = {}
    edge_containers = {etype: [] for etype in [
        ("user", "has", "opioid_level"),
        ("user", "has", "habit"),
        ("user", "has", "health_condition"),
        ("health_condition", "need", "nutrition_tag"),
        ("user", "has", "poverty_condition"),
        ("poverty_condition", "need", "price_tag"),
        ("user", "eat", "food"),
        ("food", "belongs_to", "category"),
        ("food", "has", "ingredient"),
        ("food", "contains", "nutrition_tag"),
        ("food", "cost", "price_tag"),
    ]}

    # Habit
    unique_habits = habits["habitID"].astype(str).unique().tolist()
    data["habit"].node_id = unique_habits
    data["habit"].x = torch.ones((len(unique_habits), 1))

    # Health condition
    data["health_condition"].node_id = list(reference_dict.keys())
    data["health_condition"].x = torch.ones((len(reference_dict), 1))

    # Nutrition tag
    unique_tags = list({tag for tags in reference_dict.values() for tag in tags} | set(food_primary_nutrition_tags))
    data["nutrition_tag"].node_id = unique_tags
    data["nutrition_tag"].x = torch.ones((len(unique_tags), 1))

    # Poverty condition & price tags
    data["poverty_condition"].node_id = ["poverty/food_insecurity"]
    data["poverty_condition"].x = torch.ones((1, 1))

    data["price_tag"].node_id = ["Low_price", "Medium_price", "High_price"]
    data["price_tag"].x = torch.ones((3, 1))

    for _, row in tqdm(users.iterrows(), total=len(users), desc="Building hetero graph"):
        uid = int(row["SEQN"])
        opioid_level = int(row["opioid_label"])

        # user
        if "x" not in data["user"]:
            data["user"].x, data["user"].node_id = [], []
        data["user"].x.append([1.0])
        data["user"].node_id.append(uid)
        user_idx = len(data["user"].x) - 1
        user_labels[user_idx] = opioid_level

        # opioid level
        if "x" not in data["opioid_level"]:
            data["opioid_level"].x, data["opioid_level"].node_id = [], []
        if opioid_level not in data["opioid_level"].node_id:
            data["opioid_level"].x.append([1.0])
            data["opioid_level"].node_id.append(opioid_level)
        ol_idx = data["opioid_level"].node_id.index(opioid_level)
        edge_containers[("user", "has", "opioid_level")].append((user_idx, ol_idx))

        user_habits = habits.loc[habits["SEQN"] == uid, "habitID"].astype(str).tolist()
        for h in user_habits:
            if h in data["habit"].node_id:
                h_idx = data["habit"].node_id.index(h)
                edge_containers[("user", "has", "habit")].append((user_idx, h_idx))

        # health_condition
        for status, nutrition_tags in reference_dict.items():
            if status in users.columns and row[status] == 1:
                s_idx = data["health_condition"].node_id.index(status)
                edge_containers[("user", "has", "health_condition")].append((user_idx, s_idx))
                for nut in nutrition_tags:
                    if nut in data["nutrition_tag"].node_id:
                        n_idx = data["nutrition_tag"].node_id.index(nut)
                        edge_containers[("health_condition", "need", "nutrition_tag")].append((s_idx, n_idx))

        # poverty condition
        if (row.get("food_insecurity", 0) == 1) or (row.get("low_income", 0) == 1):
            edge_containers[("user", "has", "poverty_condition")].append((user_idx, 0))
            edge_containers[("poverty_condition", "need", "price_tag")].append((0, 0))

        # food relations
        for fid in fooduser.loc[fooduser["SEQN"] == uid, "food_id"].tolist():
            food = foods[foods["food_id"] == fid]
            if food.empty:
                continue
            if "x" not in data["food"]:
                data["food"].x, data["food"].node_id = [], []
            fid_str = str(fid)
            if fid_str not in data["food"].node_id:
                data["food"].x.append([1.0])
                data["food"].node_id.append(fid_str)
            f_idx = data["food"].node_id.index(fid_str)
            edge_containers[("user", "eat", "food")].append((user_idx, f_idx))

            for _, ing in fndds_ingredients[fndds_ingredients["food_id"] == fid].iterrows():
                cat_id = int(ing["WWEIA_id"])
                if "x" not in data["category"]:
                    data["category"].x, data["category"].node_id = [], []
                if cat_id not in data["category"].node_id:
                    data["category"].x.append([1.0])
                    data["category"].node_id.append(cat_id)
                c_idx = data["category"].node_id.index(cat_id)
                edge_containers[("food", "belongs_to", "category")].append((f_idx, c_idx))

                ing_id = int(ing["ingredient_id"])
                if "x" not in data["ingredient"]:
                    data["ingredient"].x, data["ingredient"].node_id = [], []
                if ing_id not in data["ingredient"].node_id:
                    data["ingredient"].x.append([1.0])
                    data["ingredient"].node_id.append(ing_id)
                i_idx = data["ingredient"].node_id.index(ing_id)
                edge_containers[("food", "has", "ingredient")].append((f_idx, i_idx))

            for col in food_primary_nutrition_tags:
                if col in food.columns and food[col].iloc[0] == 1:
                    n_idx = data["nutrition_tag"].node_id.index(col)
                    edge_containers[("food", "contains", "nutrition_tag")].append((f_idx, n_idx))

            for pl in ["Low_price", "Medium_price", "High_price"]:
                if pl in food.columns and food[pl].iloc[0] == 1:
                    p_idx = data["price_tag"].node_id.index(pl)
                    edge_containers[("food", "cost", "price_tag")].append((f_idx, p_idx))

    # tensorize
    for ntype in data.node_types:
        if isinstance(data[ntype].x, list):
            data[ntype].x = torch.tensor(data[ntype].x, dtype=torch.float)

    for etype, pairs in edge_containers.items():
        if pairs:
            edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
            data[etype].edge_index = torch.unique(edge_index, dim=1)

    data["user"].y = torch.tensor([user_labels[i] for i in range(len(user_labels))], dtype=torch.long)
    return data


users = pd.read_csv(USER_FILE)
foods = pd.read_csv(FOOD_FILE)
fndds_ingredients = pd.read_csv(FNDDS_FILE)
habits = pd.read_csv(HABIT_FILE)
fooduser = pd.read_csv(FOODUSER_FILE)

graph = build_big_hetero_graph(users, foods, fndds_ingredients, habits,
                               fooduser, reference_dict, food_primary_nutrition_tags)
torch.save(graph, "./graph/hetero_graph.pt")

print(graph)

for ntype in graph.node_types:
    print(f"{ntype} nodes count: {graph[ntype].num_nodes}")

for etype in graph.edge_types:
    print(f"{etype} edges count: {graph[etype].edge_index.size(1)}")

print("users count:", graph["user"].num_nodes)
print("label distribution:", torch.bincount(graph["user"].y))

