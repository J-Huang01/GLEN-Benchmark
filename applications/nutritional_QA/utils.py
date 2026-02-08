import re
import networkx as nx

import os
from openai import OpenAI

def get_openrouter_client(api_key: str | None = None, title: str = "BLEN QA Bench"):

    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY")

    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "https://your.domain-or-localhost",  # 没网站就用 localhost
            "X-Title": title,
        },
    )
def convert_to_txt(path):
    full_path = path[:1]
    for i in range(1, len(path) - 1):
        if i % 2 == 1:
            full_path.append(path[i])
        else:
            full_path.extend([path[i], ',', path[i]])
    full_path.append(path[-1])
    return ' '.join(full_path)


def convert_to_sg(graph, path_list):
    """
    Generate subgraphs based on the list of paths and merge these subgraphs into a new subgraph.

    Args:
        graph (nx.Graph or nx.DiGraph): The original graph.
        path_list (list of list): Each path is a collection of node attributes and relationships.

    Returns:
        nx.DiGraph: The merged subgraph.
    """
    merged_graph = nx.DiGraph()  # Used to store the merged subgraph

    for path in path_list:
        # Create a subgraph for the current path
        sg = graph.copy()
        
        # Remove nodes not included in the path
        nodes_to_remove = [node for node in sg.nodes() if str(sg.nodes[node]['attr']) not in path]
        sg.remove_nodes_from(nodes_to_remove)

        # Remove edges not included in the path
        edges_to_remove = [
            (u, v) for u, v, attr in sg.edges(data=True)
            if str(sg.nodes[u]['attr']) not in path
            or str(sg.nodes[v]['attr']) not in path
            or attr['relationship'] not in path
        ]
        sg.remove_edges_from(edges_to_remove)

        # Merge the nodes and edges of the current subgraph into the merged graph
        merged_graph.add_nodes_from(sg.nodes(data=True))
        merged_graph.add_edges_from(sg.edges(data=True))

    return merged_graph

def find_relations(graph, path):
    candidate_path_list = []
    last_node = path[-1]
    
     
    for src, tgt, data in graph.edges(data=True):
        src_attr = str(graph.nodes[src].get('attr', src))
        tgt_attr = str(graph.nodes[tgt].get('attr', tgt))
        relationship = data.get('relationship', 'unknown')

        is_connected = (src_attr == last_node or tgt_attr == last_node)

        both_in_path = (src_attr in path and tgt_attr in path)
        
        if is_connected and not both_in_path:
            candidate_path = path + [relationship]
            if candidate_path not in candidate_path_list:
                candidate_path_list.append(candidate_path)
                
    return candidate_path_list

def prune_relations(client, path_list, question, model_name, width):
    if len(path_list) <= width:
        return path_list

    reasoning_path_lines = [f"{i+1}. {convert_to_txt(path)}." for i, path in enumerate(path_list)]
    reasoning_path_text = "\n".join(reasoning_path_lines)

    messages = [
        {
            "role": "system",
            "content": (
                f"Identify the top-{width} reasoning paths that are most likely to lead to the answer for the query. "
                "Respond with the indices of the reasoning paths, starting from 1, separated by commas (e.g., 1,2,5). "
                "Include nothing else in your response."
            ),
        },
        {
            "role": "user",
            "content": (
                f"The query is: {question}\n\n"
                f"The reasoning paths are:\n{reasoning_path_text}\n\n"
                f"Your selected top-{width} reasoning paths are:"
            ),
        },
    ]

    answer = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0
    ).choices[0].message.content

    indices = [int(x) - 1 for x in re.findall(r"\d+", answer)]
    indices = [i for i in indices if 0 <= i < len(path_list)][:width]
    if not indices:
        return path_list[:width]
    return [path_list[i] for i in indices]


def find_entities(graph, reasoning_path):
    """
    Expand reasoning path by adding the target entity.
    
    Args:
        reasoning_path: ['user', 'has']  # 路径以关系结尾
    
    Returns:
        List of extended paths with entities added
    """
    candidate_path_list = []

    if not reasoning_path or len(reasoning_path) < 2:
        return [reasoning_path]
    
    node_name = reasoning_path[-2]
    edge_name = reasoning_path[-1]
    
    for src, tgt, data in graph.edges(data=True):
        src_attr = str(graph.nodes[src].get('attr', src))
        tgt_attr = str(graph.nodes[tgt].get('attr', tgt))
        relationship = data.get('relationship', 'unknown')

        if src_attr == node_name and relationship == edge_name:
            if tgt_attr not in reasoning_path:
                new_path = reasoning_path + [tgt_attr]
                if new_path not in candidate_path_list:
                    candidate_path_list.append(new_path)

        elif tgt_attr == node_name and relationship == edge_name:
            if src_attr not in reasoning_path:
                new_path = reasoning_path + [src_attr]
                if new_path not in candidate_path_list:
                    candidate_path_list.append(new_path)

    return candidate_path_list if candidate_path_list else [reasoning_path]

def prune_entities(client, path_list, question, model_name, width):
    if len(path_list) <= width:
        return path_list

    reasoning_path_lines = [f"{i+1}. {convert_to_txt(path)}." for i, path in enumerate(path_list)]
    reasoning_path_text = "\n".join(reasoning_path_lines)

    messages = [
        {
            "role": "system",
            "content": (
                f"Identify the top-{width} reasoning paths that are most likely to lead to the answer for the query. "
                "Respond with the indices of the reasoning paths, starting from 1, separated by commas (e.g., 1,2,5). "
                "Include nothing else in your response."
            ),
        },
        {
            "role": "user",
            "content": (
                f"The query is: {question}\n\n"
                f"The reasoning paths are:\n{reasoning_path_text}\n\n"
                f"Your selected top-{width} reasoning paths are:"
            ),
        },
    ]

    answer = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0
    ).choices[0].message.content

    indices = [int(x) - 1 for x in re.findall(r"\d+", answer)]
    indices = [i for i in indices if 0 <= i < len(path_list)][:width]
    if not indices:
        return path_list[:width]
    return [path_list[i] for i in indices]


def generate_paragraph_cot_bag(textualized_triplets):
    """
    Convert textualized triplets into a paragraph format for CoT_BaG.

    Args:
        textualized_triplets (str): A single string of textualized triplets.

    Returns:
        str: Paragraph format of the input triplets.
    """
    nodes = set()
    edges = []

    valid_relationships = ["belongs to", "has", "contains", "match", "contradict", "need"]

    triplet_list = textualized_triplets.strip().strip("()").split("), (")
    
    for triplet in triplet_list:
        try:
            for relationship in valid_relationships:
                if f" {relationship} " in triplet:
                    source, target = triplet.split(f" {relationship} ", 1)
                    source = source.strip()
                    target = target.strip()

                    nodes.update([source, target])
                    edges.append(f'an edge between node "{source}" directed to node "{target}" with attribute "{relationship}"')
                    break
            else:
                print(f"Warning: Relationship not found in triplet '{triplet}'")
        except Exception as e:
            print(f"Error processing triplet '{triplet}': {e}")

    # Convert nodes and edges to paragraph format
    node_list = ", ".join(f'"{node}"' for node in nodes)
    nodes_edges_list = ", ".join(edges)

    paragraph = (
        f"""
        Here is the description of the graph: 
        This is the list of edges: {nodes_edges_list}.
        Let's first construct a graph with the given nodes and edges. Let's think step by step. Determine the healthiness of the food by traversing the graph and determining the nutritional properties of the food, then compare them to the health status, dietary need and habits of the user. Do not be too strict with your criteria, only focus on a few main nutritional tags that strongly indicate its healthiness or unhealthiness to the particular diet or health status the user has. Some nutritional tags might not be as important in determining healthiness. Do not regard a food as unhealthy just because it has some negative nutrition tags because these tags might not always be important in the user's case.
        Provide the output adhering to the following guideline.
        """    
    )

    return paragraph

import pandas as pd
import random 
import copy


def concat_data_across_years(data_type, file_code, years, year_char):
    """
    This function concat data across years.

    :param data_type: The NHANES data type (demographic, dietary, examination, laboratory, questionnaire)
    :param file_code: the NHANES code of the file.
    :param years: A list of strings identifies which year the data is from. E.g. ['0102', '0304', ..., '1314']
    :param year_char: The char NHANES data used to identify years. This char attaches to the file names as suffix.
        For 01-02 data, the char is B; For 03-04 data, the char is C.
    :return: Returns a pandas dataframe of the concatenated data
    """
    df = pd.DataFrame()
    root_path = '../data/'
    for year in years:
        if year == '1720':
            path = root_path + year + '/' + data_type + '/P_' + file_code + '.XPT'
        else:
            path = root_path + year + '/' + data_type + '/' + file_code + '_' + year_char + '.XPT'

        df_temp = pd.read_sas(path, encoding='ISO-8859-1')
        year_char = chr(ord(year_char) + 1)
        df_temp['years'] = year

        if df.empty:
            df = df_temp.copy()
        else:
            df = pd.concat([df, df_temp])

    df.replace(5.397605346934028e-79, 0, inplace=True)
    return df


def merge_with_or(df1, df2):
    """
    Merges two DataFrames on a specified key(s) and combines shared columns using an 'OR' relation. This function is 
    used when merging nutrition tags because we want to combine the tags from different sources using an 'OR' logic.
    
    Parameters:
    - df1, df2: DataFrames to be merged.
    
    Returns:
    - A merged DataFrame with combined shared columns using an 'OR' logic.
    """
    # Merge the DataFrames
    merged_df = pd.merge(df1, df2, left_index=True, right_index=True, how='left', suffixes=('_df1', '_df2'))
    merged_df = merged_df.fillna(0).astype(int)
    
    # Find shared columns, excluding the key(s) used for merging
    shared_columns = set(df1.columns) & set(df2.columns)
    
    for col in shared_columns:
        col_df1 = f'{col}_df1'
        col_df2 = f'{col}_df2'

        # Apply 'OR' operation for the shared column and assign it to the merged DataFrame
        # Only df2 contains NaN values, so we need to fill them with False before converting to int
        merged_df[col] = (merged_df[col_df1] | merged_df[col_df2].fillna(False).astype(int))
        
        # Drop the original columns from the merge
        merged_df.drop(columns=[col_df1, col_df2], inplace=True)
    
    return merged_df


def convert_tags(row, primary_nutrition_tags, nutrition_dict={}):
    """
    Converts high/low nutrition tags to a uniform dictionary format.
    Returns a dictionary with tag names as keys and values as -1 (low) or 1 (high).
    """
    for nutrition_tag in primary_nutrition_tags:
        if row[nutrition_tag] == 1:
            if 'low' in nutrition_tag:
                nutrition_dict[nutrition_tag[4:]] = -1
            else:
                nutrition_dict[nutrition_tag[5:]] = 1
    
    return nutrition_dict


def generate_pairs(food_nutrition_dict, nutrition_list, user_list, count, level, food_id):
    """
    Generates user-food pairs by selecting users who match specific nutrition tags associated with a food item, 
    according to a specified difficulty level.

    Parameters:
    - food_nutrition_dict: Dictionary with nutrition tags and values for the target food.
    - nutrition_list: List of nutrition tags relevant to the matching.
    - user_list: List of users with their nutrition tags.
    - count: The number of pairs to generate.
    - level: Difficulty level ('easy', 'medium', 'hard') controlling tag matching criteria.
    - food_id: Identifier for the food, also used as a seed for randomness.

    Returns:
    - A list of deduplicated user dictionaries that match the nutrition criteria based on the selected difficulty level.
    """
    pair_results = []
    added_users = set()  # Track users to avoid duplication
    loop_seed = food_id
    while True:
        loop_seed += 1
        random.seed(loop_seed)
        random.shuffle(nutrition_list)

        # Set number of tags in common based on difficulty level
        if level == 'easy':
            num_tag_in_common = 1
        else:
            num_tag_in_common = random.randint(2, len(nutrition_list))

        selected_nutrition = nutrition_list[:num_tag_in_common]
        avoid_nutrition = nutrition_list[num_tag_in_common:]

        # Shuffle user_list in place
        random.shuffle(user_list)
        for user in user_list:
            user_id = user.get('user_id')  # Assuming each user has a unique identifier
            if user_id in added_users:
                continue
            # Check if user only has selected nutrition tags
            if any(nutrition in user for nutrition in avoid_nutrition):
                continue
            if not all(nutrition in user for nutrition in selected_nutrition):
                continue

            # Specific condition for 'medium' and 'hard'
            if level in ['medium', 'hard']:
                tag_match_count = sum(
                    1 if food_nutrition_dict[nutrition] == user[nutrition] else -1
                    for nutrition in selected_nutrition
                )
                if (level == 'medium' and abs(tag_match_count) != len(selected_nutrition)) or \
                   (level == 'hard' and abs(tag_match_count) == len(selected_nutrition)):
                    continue

            pair_results.append(user)
            added_users.add(user_id)
            break

        if len(pair_results) == count or loop_seed > 50 + food_id:
            break

    return pair_results


def generate_answer(food_tag, user):
    """
    Generates answers based on the compatibility between a food's nutrition tags and a user's preferences.

    Parameters:
    - food_tag: Dictionary containing nutrition tags and their levels ('high' or 'low') for the food.
    - user: Dictionary containing user-specific nutrition preferences.

    Returns:
    - answer_easy: A simple 'Yes' or 'No' indicating compatibility at a high level.
    - answer_medium: A string listing the nutrition tags involved in the decision-making process.
    - answer_hard: A detailed explanation of why the food is or isn't compatible with the user.
    """
    # List to hold positive reasons and negative reasons 
    reasons_for_yes = []  
    reasons_for_no = []   
    used_nutrition_tags = []  # Tracks nutrition tags considered in the decision

    # Iterate through user's nutrition preferences, skipping the first key (e.g., user_id)
    for nutrition in list(user.keys())[1:]:
        if nutrition in list(food_tag.keys()):  
            level = 'high' if food_tag[nutrition] == 1 else 'low'
            used_nutrition_tags.append(level + '_' + nutrition)  
            
            # Append matching or mismatching reasons
            if user[nutrition] == food_tag[nutrition]:
                reasons_for_yes.append(f"{level} in {nutrition}")
            else:
                reasons_for_no.append(f"{level} in {nutrition}")

    # Generate the detailed "hard" answer based on majority matching or non-matching reasons.
    # Note that only hard questions can have conflicting reasons. This is how we determine the final answer.
    if len(reasons_for_no) >= len(reasons_for_yes):
        answer_hard = 'No, because the food is '
        for reason in reasons_for_no:
            answer_hard += reason + ', '  
        answer_hard = answer_hard[:-2] + '. '  
    else:
        answer_hard = 'Yes, because the food is '
        for reason in reasons_for_yes:
            answer_hard += reason + ', '  
        answer_hard = answer_hard[:-2] + '. '  

    answer_easy = 'Yes' if 'Yes' in answer_hard else 'No'
    answer_medium = ', '.join(used_nutrition_tags)

    return answer_easy, answer_medium, answer_hard


def generate_graph(food_id, user_id, food_info, food_ingredients, user_info, user_habits, food_primary_nutrition_tags, reference_dict):
    """
    Generates a graph structure representing relationships between a user, food, and their attributes.

    Parameters:
    - food_id: The ID of the food item.
    - user_id: The ID of the user.
    - food_info: DataFrame containing food attributes.
    - user_info: DataFrame containing user attributes.
    - food_primary_nutrition_tags: List of primary nutrition tags for foods.
    - reference_dict: Dictionary mapping user statuses to nutrition tags.

    Returns:
    - node_list: List of nodes in the graph, each represented as [node_id, {'name': str, 'attr': value}].
    - edge_list: List of edges in the graph, each represented as [node_id, relation, node_id].
    """
    node_list, edge_list  = [], []  
    node_id = 2  # Start assigning node IDs from 2 (0: user, 1: food)

    # Add the food node
    food = food_info[food_info['food_id'] == food_id]
    node_list.append([1, {'name': int(food_id), 'attr': food['food_desc'].item()}])  # Node ID 1 is the food

    # Add the food category node
    food_category = food['WWEIA_desc'].iloc[0]
    node_list.append([node_id, {'name': 'category', 'attr': food_category}])
    edge_list.append([1, 'belongs to', node_id])
    node_id += 1
    
    # Add the food ingredient nodes
    food_ingredients = food_ingredients[food_ingredients['food_id'] == food_id]
    food_ingredients = food_ingredients['ingredient_desc'].tolist()
    for ingredient in food_ingredients:
        node_list.append([node_id, {'name': 'ingredient', 'attr': ingredient}])
        edge_list.append([1, 'has', node_id])
        node_id += 1

    # Add nutrition tags for the food
    for column in food_primary_nutrition_tags:
        if food[column].item() == 1:
            node_list.append([node_id, {'name': 'food_nutrition_tag', 'attr': column}])
            edge_list.append([1, 'contains', node_id])  # Food has this nutrition tag
            node_id += 1

    # Add the user node
    node_list.append([0, {'name': int(user_id), 'attr': 'user'}])  # Node ID 0 is the user
    user = user_info[user_info['SEQN'] == user_id]

    # Add the user habits
    habit_list = user_habits[user_habits['SEQN'] == user_id]['habitDesc'].tolist()
    for habit in habit_list:
        node_list.append([node_id, {'name': 'dietary habit', 'attr': habit}])
        edge_list.append([0, 'has', node_id])
        node_id += 1

    user_nutrition_tag_map = {}
    # Add user statuses and match them to nutrition tags
    for column, nutrition_tags in reference_dict.items():
        if user[column].item() == 1:  # If the user has this status
            status_node_id = node_id
            node_list.append([node_id, {'name': 'status', 'attr': column}])
            edge_list.append([0, 'has', node_id])  # User has this status
            node_id += 1

            # Match user statuses to food nutrition tags or create new user nutrition tag nodes
            for nutrition_tag in nutrition_tags:
                level, nutrition = nutrition_tag.split('_', maxsplit=1)
                # Check for matching food nutrition tags
                match_found = False
                for node in node_list:
                    if node[1]['name'] == 'food_nutrition_tag':
                        if nutrition in node[1]['attr']:
                            if level in node[1]['attr']:
                                edge_list.append([status_node_id, 'match', node[0]])
                            else:
                                edge_list.append([status_node_id, 'contradict', node[0]])
                            match_found = True
                            break

                if not match_found:
                    if nutrition_tag in user_nutrition_tag_map:
                        existing_node_id = user_nutrition_tag_map[nutrition_tag]
                        edge_list.append([status_node_id, 'need', existing_node_id])
                    else:
                        node_list.append([node_id, {'name': 'user_nutrition_tag', 'attr': nutrition_tag}])
                        edge_list.append([status_node_id, 'need', node_id])
                        user_nutrition_tag_map[nutrition_tag] = node_id  # 记录映射
                        node_id += 1

    return node_list, edge_list


def remove_implicit_tags(row, reference_dict):
    '''
    The original user_tagging.csv contains nutrition tags that are not infered from statuses we will use in the graph.
    Returns a filtered user row with only explicit tags.
    '''
    new_row = copy.deepcopy(row)

    # Set values to 0 except SEQN
    for key in row.keys()[1:]:
        new_row[key] = 0

    # Only pass explicit tags to new_row
    for status in reference_dict.keys():
        if row[status] == 1:
            new_row[status] = 1
            for tag in reference_dict[status]:
                new_row[tag] = 1

    return new_row