import os
import pandas as pd
import openai
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
import json, re


def is_item_data(data):
    if "item_category" in data and "title" in data:
        return True
    elif "channel_category" in data and "channel_name" in data:
        return False


class TextEmbedder:
    def __init__(self, api_key=None, model_name='text-embedding-3-small'):
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        self.embedder = OpenAIEmbeddings(model=model_name)

    def get_text_embedding(self, text):
        return self.embedder.embed_query(text)

    def get_texts_embedding(self, texts):
        return self.embedder.embed_documents(texts)


# Step 1: OpenAI API 키 설정
def setup_openai_api():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")


# Step 2: 데이터 로드
def load_data(creator_path, item_path):
    creators = pd.read_csv(creator_path)
    items = pd.read_csv(item_path)
    return creators, items


# Stage1
# 그래프 생성 함수
def generate_graph(creators_df, items_df, embedder):
    # Create embeddings f
    creator_embeddings = [embedder.get_text_embedding(text) for text in creators_df['channel_category']]
    item_embeddings = [embedder.get_text_embedding(text) for text in items_df['item_category']]
    similarity_matrix = cosine_similarity(creator_embeddings, item_embeddings)

    # Generate pseudo graph connections
    connections = {}
    for i, creator in creators_df.iterrows():
        creator_id = creator['creator_id']
        connections[creator_id] = {"direct": [], "indirect": []}
        for j, item in items_df.iterrows():
            if similarity_matrix[i, j] > 0.6:  # Direct connection threshold
                connections[creator_id]["direct"].append(item['item_id'])
            elif similarity_matrix[i, j] > 0.4:  # Indirect connection threshold
                connections[creator_id]["indirect"].append(item['item_id'])
    return connections


# Cold-start 후보군 생성 함수
def candidates_with_graph(data, connections, creators_df, items_df, embedder, top_k=10):
    if is_item_data(data):  # 새로운 데이터가 'item'일 경우
        print("\nProcessing as item data...")

        # 'item' 데이터를 기반으로 추천할 'user' 후보 생성
        candidates = candidates_for_item(data, connections, creators_df, embedder, top_k)
        for candidate in candidates:
            if 'channel_name' not in candidate:
                creator_row = creators_df.loc[creators_df['creator_id'] == candidate['creator_id']]
                print("Creator Row for ID:", candidate['creator_id'], creator_row)
                if not creator_row.empty:
                    candidate['channel_name'] = creator_row['channel_name'].values[0]

    else:  # 새로운 데이터가 'user'일 경우
        print("\nProcessing as user data...")
        candidates = candidates_for_user(data, connections, items_df, embedder, top_k)
        for candidate in candidates:
            if 'title' not in candidate:
                item_row = items_df.loc[items_df['item_id'] == candidate['item_id']]
                print("Item Row for ID:", candidate['item_id'], item_row)
                if not item_row.empty:
                    candidate['title'] = item_row['title'].values[0]
    return candidates


def candidates_for_item(item_data, connections, creators_df, embedder, top_k=20):
    item_embedding = embedder.get_text_embedding(item_data['item_category'])

    # Calculate similarity
    creator_embeddings = embedder.get_texts_embedding(creators_df['channel_category'].tolist())
    similarity_scores = cosine_similarity([item_embedding], creator_embeddings).flatten()

    # Rank and format recommendations
    top_k_indices = similarity_scores.argsort()[-top_k:][::-1]
    recommendations = []
    for idx in top_k_indices:
        creator = creators_df.iloc[idx]
        recommendations.append({
            "creator_id": creator['creator_id'],
            "channel_name": creator.get('channel_name', None),
            "channel_category": creator['channel_category'],
            "subscribers": creator['subscribers']
        })

    return recommendations


def candidates_for_user(user_data, connections, items_df, embedder, top_k=20):
    user_embedding = embedder.get_text_embedding(user_data['channel_category'])
    item_embeddings = embedder.get_texts_embedding(items_df['item_category'].tolist())
    similarity_scores = cosine_similarity([user_embedding], item_embeddings).flatten()
    top_k_indices = similarity_scores.argsort()[-top_k:][::-1]

    recommendations = []
    for idx in top_k_indices:
        item = items_df.iloc[idx]
        recommendations.append({
            "item_id": item['item_id'],
            "title": item['title'],
            "item_category": item['item_category'],
            "media_type": item['media_type'],
            "score": item.get('score', 0),
            "item_content": item.get('item_content', "No content available")
        })
    return recommendations


# Stage2
class LLMCandidateRanker:
    def __init__(self, api_key, model_name="gpt-4", temperature=0.5):
        self.chat_model = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=api_key
        )

    def generate_graph_context(self, connections, user_or_item_id, is_item):
        if user_or_item_id not in connections:
            print(f"Warning: {user_or_item_id} not found in connections!")
            return "Direct Connections: []\nIndirect Connections: []"

        # Retrieve connections
        graph_data = connections.get(user_or_item_id, {})
        direct = graph_data.get("direct", [])
        indirect = graph_data.get("indirect", [])

        return (
            f"Direct Connections: {direct}\n"
            f"Indirect Connections: {indirect}"
        )

    def generate_personalized_prompt(self, data, candidates, connections, is_item, top_k=20):
        metadata = (
            f"Category: {data['item_category']} | Media Type: {data['media_type']} | Score: {data['score']}"
            if is_item
            else f"Channel Category: {data['channel_category']} | Subscribers: {data['subscribers']}"
        )

        graph_context = self.generate_graph_context(connections, data['item_id' if is_item else 'creator_id'], is_item)

        candidates_str = "\n".join(
            [
                f"({chr(65 + idx)}) "
                f"{candidate.get('title', 'N/A') if not is_item else candidate.get('channel_name', 'N/A')}: "
                f"{candidate.get('item_category', 'N/A') if not is_item else candidate.get('channel_category', 'N/A')}, "
                f"{'Short' if not is_item and candidate.get('media_type', 'N/A') == 'short' else 'Long'}, "
                f"Score: {candidate.get('score', 'N/A') if not is_item else candidate.get('subscribers', 'N/A')}"
                for idx, candidate in enumerate(candidates)
            ]
        )
        id_map = {chr(65 + idx): candidate['creator_id' if is_item else 'item_id'] for idx, candidate in
                  enumerate(candidates)}
        print("id_map:", id_map)
        prompt = f"""
        ### Instruction:
        Based on the {'item' if is_item else 'user'}'s metadata and graph connections, rank the following candidates. Consider all candidates provided, regardless of their category, and prioritize based on relevance and score
        ### {'Item' if is_item else 'User'} Metadata:
        {metadata}
        ### Graph Context:
        {graph_context}
        ### Candidate {'Creators' if is_item else 'Items'}:
        {candidates_str}
        ### Response:
        Return a comma-separated list of indices (A, B, C, ...) corresponding to the best candidates in order of preference. Ensure to include at least {top_k} candidates. 
        If there are fewer than {top_k} suitable candidates, include as many as possible.
        Example Response: A, B, C, D
        """
        return prompt.strip(), id_map

    def rank_candidates(self, prompt):
        messages = [
            SystemMessage(content="You are an assistant that ranks recommendation candidates."),
            HumanMessage(content=prompt)
        ]

        response = self.chat_model.invoke(messages)

        ranked_indices = response.content.strip().split(", ")
        return ranked_indices

    def re_rank(self, data, candidates, connections, is_item=True):
        print("\n--- Re-ranking Candidates ---")
        prompt, id_map = self.generate_personalized_prompt(data, candidates, connections, is_item, top_k=20)
        ranked_indices = self.rank_candidates(prompt)
        print("Ranked Indices:", ranked_indices)
        ranked_ids = [id_map[idx] for idx in ranked_indices if idx in id_map]

        ranked_candidates = [candidate for candidate in candidates if
                             candidate['creator_id' if is_item else 'item_id'] in ranked_ids]
        print("Ranked Candidates:", ranked_candidates)
        return ranked_candidates, id_map


def update_connections_with_creator(connections, creators_df, items_df, embedder, new_creator):
    creator_id = new_creator['creator_id']
    connections[creator_id] = {"direct": [], "indirect": []}

    new_creator_embedding = embedder.get_text_embedding(new_creator['channel_category'])
    item_embeddings = embedder.get_texts_embedding(items_df['item_category'].tolist())
    similarity_scores = cosine_similarity([new_creator_embedding], item_embeddings).flatten()

    for idx, score in enumerate(similarity_scores):
        item_id = items_df.iloc[idx]['item_id']
        if score > 0.6:
            connections[creator_id]["direct"].append(item_id)
        elif score > 0.4:
            connections[creator_id]["indirect"].append(item_id)


def update_connections_with_item(connections, creators_df, items_df, embedder, new_item):
    item_id = new_item['item_id']

    new_item_embedding = embedder.get_text_embedding(new_item['item_category'])
    creator_embeddings = embedder.get_texts_embedding(creators_df['channel_category'].tolist())
    similarity_scores = cosine_similarity([new_item_embedding], creator_embeddings).flatten()

    for idx, score in enumerate(similarity_scores):
        creator_id = creators_df.iloc[idx]['creator_id']
        if score > 0.6:
            connections[creator_id]["direct"].append(item_id)
        elif score > 0.4:
            connections[creator_id]["indirect"].append(item_id)


# 공통 로직 분리
def generate_candidates(new_data, creators_df, items_df, embedder, connections, top_k, is_item):
    if is_item:
        update_connections_with_item(connections, creators_df, items_df, embedder, new_data)
    else:
        update_connections_with_creator(connections, creators_df, items_df, embedder, new_data)

    candidates = candidates_with_graph(new_data, connections, creators_df, items_df, embedder, top_k * 2)
    return candidates


def rank_candidates_with_llm(new_data, ranked_candidates, connections, llm_ranker, id_map, is_item, top_k, creators_df,
                             items_df):
    candidates_str = "\n".join([
        f"({chr(65 + idx)}) {candidate.get('channel_name', 'N/A') if is_item else candidate.get('title', 'N/A')}: "
        f"{candidate.get('channel_category', 'N/A') if is_item else candidate.get('item_category', 'N/A')}, "
        f"Score: {candidate.get('subscribers', 'N/A') if is_item else candidate.get('score', 'N/A')}"
        for idx, candidate in enumerate(ranked_candidates)
    ])

    graph_context = connections.get(new_data['item_id' if is_item else 'creator_id'], {})
    if not graph_context:
        graph_context = {"direct": [], "indirect": []}

    prompt = f"""
    ### Instruction:
    Based on the provided information, rank the following candidates in order of preference and return the results in a structured JSON format.

    ### New Data:
    {new_data}

    ### Graph Context:
    Direct Connections: {graph_context.get('direct', [])}
    Indirect Connections: {graph_context.get('indirect', [])}

    ### Candidates:
    {candidates_str}

    ### Response:
    Return a JSON array of the top {top_k} candidates. Each candidate should have the following format:
    If item_id = True,
    [
        {{
            "creator_id": <id>,   
            "channel_category": "<category>", 
            "channel_name": "<name>",         
            "subscribers": <int>            
        }},

    ]  

    If item_id = False,

    [
        {{
            "item_id": <id>,  
            "title": "<name>", 
            "item_category": "<item_category>", 
            "media_type" : <media_type>
            "score": <score>  
            "item_content": <item_content>           
        }},
    ]  
    Only include the top {top_k} candidates.

    """

    # LLM 호출
    messages = [
        SystemMessage(content="You are an assistant that ranks recommendation candidates."),
        HumanMessage(content=prompt)
    ]
    response = llm_ranker.chat_model.invoke(messages)

    # JSON 추출 및 파싱
    try:
        json_match = re.search(r"\[\s*{.*?}\s*\]", response.content.strip(), re.DOTALL)

        json_content = json_match.group(0)
        recommendations = json.loads(json_content)


    except Exception as e:
        print("Error parsing LLM response:", e)
        print("--- Raw LLM Response ---")
        print(response.content.strip())
        return []

    # Title 또는 Channel Name을 기준으로 최종 추천 데이터 생성
    final_recommendations = []
    for recommendation in recommendations:
        if is_item:
            # channel_name을 기준으로 creators_df에서 데이터 매핑
            channel_name = recommendation.get('channel_name')
            matching_creator = creators_df[creators_df['channel_name'] == channel_name]
            if not matching_creator.empty:
                creator = matching_creator.iloc[0].to_dict()
                final_recommendations.append({
                    "creator_id": creator['creator_id'],
                    "channel_category": creator['channel_category'],
                    "channel_name": creator['channel_name'],
                    "subscribers": int(creator['subscribers'])
                })

        else:
            # title을 기준으로 items_df에서 데이터 매핑
            title = recommendation.get('title')
            matching_item = items_df[items_df['title'] == title]
            if not matching_item.empty:
                item = matching_item.iloc[0].to_dict()
                final_recommendations.append({
                    "item_id": item['item_id'],
                    "item_category": item['item_category'],
                    "title": item['title'],
                    "score": item['score'],
                    "media_type": item['media_type'],
                    "item_content": item.get('item_content', "No content available")
                })

    return final_recommendations


# 추천 함수
def recommend_for_new_creator(new_creator_data, creators_df, items_df, embedder, connections, llm_ranker, top_k=10):
    print("\n--- Generating Candidates for New Creator ---")
    candidates = generate_candidates(new_creator_data, creators_df, items_df, embedder, connections, top_k,
                                     is_item=False)
    ranked_candidates, id_map = llm_ranker.re_rank(new_creator_data, candidates, connections, is_item=False)
    final_recommendations = rank_candidates_with_llm(
        new_creator_data, ranked_candidates, connections, llm_ranker, id_map, is_item=False, top_k=top_k,
        creators_df=creators_df, items_df=items_df
    )

    print("\n추천 아이템 목록:")
    formatted_recommendations = [
        {
            "item_id": rec["item_id"],
            "title": rec["title"],
            "item_category": rec["item_category"],
            "media_type": rec["media_type"],
            "score": rec["score"],
            "item_content": rec.get("item_content", "No content available")
        }
        for rec in final_recommendations
    ]
    print(formatted_recommendations)
    return formatted_recommendations


def recommend_for_new_item(new_item_data, creators_df, items_df, embedder, connections, llm_ranker, top_k=10):
    print("\n--- Generating Candidates for New Item ---")
    candidates = generate_candidates(new_item_data, creators_df, items_df, embedder, connections, top_k, is_item=True)
    ranked_candidates, id_map = llm_ranker.re_rank(new_item_data, candidates, connections, is_item=True)
    final_recommendations = rank_candidates_with_llm(
        new_item_data, ranked_candidates, connections, llm_ranker, id_map, is_item=True, top_k=top_k,
        creators_df=creators_df, items_df=items_df
    )

    print("\n추천 사용자 목록:")
    formatted_recommendations = [
        {
            "creator_id": rec["creator_id"],
            "channel_category": rec["channel_category"],
            "channel_name": rec["channel_name"],
            "subscribers": rec["subscribers"]
        }
        for rec in final_recommendations
    ]
    print(formatted_recommendations)
    return formatted_recommendations


# 메인 함수
def main():
    # Step 1: Load OpenAI API key
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY not found. Please set the API key in your environment.")
        return

    # Step 2: Initialize embedder
    embedder = TextEmbedder(api_key=api_key, model_name="text-embedding-3-small")

    # Step 3: Load data
    creator_path = "input/Creator_random25.csv"
    item_path = "input/Item_random25.csv"
    creators_df, items_df = load_data(creator_path, item_path)

    # Step 4: Define new creator and item data
    new_creator_data = {
        'channel_category': "tech",
        'channel_name': "최마태의 POST IT",
        'subscribers': 263000
    }

    new_item_data = {
        'title': "바밤바를 뛰어넘는 밤 맛 과자가 있을까?",
        'item_category': 'entertainment',
        'media_type': 'short',
        'score': 80,
        'item_content': '다양한 밤 맛 과자를 비교하며 맛과 질감을 리뷰하는 콘텐츠'
    }

    # Assign unique IDs to new data
    max_creator_id = creators_df['creator_id'].max()
    max_item_id = items_df['item_id'].max()
    new_creator_data['creator_id'] = max_creator_id + 1
    new_item_data['item_id'] = max_item_id + 1

    # Step 5: Generate graph
    connections = generate_graph(creators_df, items_df, embedder)

    # Step 6: Initialize LLM Candidate Ranker
    is_item = is_item_data(new_creator_data)
    print(f"\n{'Item' if is_item else 'User'} data detected. Starting LLM re-ranking...")
    ranker = LLMCandidateRanker(api_key=api_key, model_name="gpt-4", temperature=0.7)

    # Step 7: Generate recommendations
    print("\n### Recommendations for New Creator ###")
    creator_recommendations = recommend_for_new_creator(new_creator_data, creators_df, items_df, embedder, connections,
                                                        ranker, top_k=10)

    print("\n### Recommendations for New Item ###")
    item_recommendations = recommend_for_new_item(new_item_data, creators_df, items_df, embedder, connections, ranker,
                                                  top_k=10)


if __name__ == "__main__":
    main()
