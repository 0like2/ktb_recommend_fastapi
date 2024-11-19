from pydantic import BaseModel
from typing import List
from method.recommend import load_model_and_embeddings, recommend_for_new_creator, recommend_for_new_item
from schemas import CreatorRecommendRequest, ItemRecommendRequest, CreatorRecommendResponse, \
    ItemRecommendResponse
import torch
import os

# 모델 및 임베딩 로드
output_dir = "./model"
model_path = os.path.join(output_dir, "saved_model.pth")
item_emb_path = os.path.join(output_dir, "item_embeddings.pth")
graph_path = os.path.join(output_dir, "train_g.bin")

# 모델, 임베딩, 그래프를 로드합니다.
model, item_emb, graph = load_model_and_embeddings(model_path, item_emb_path, graph_path)
h_item = item_emb.weight.detach()


# 새로운 데이터로 크리에이터 -> 아이템을 10개 추천
def recommend_for_creator(request: CreatorRecommendRequest, item_data) -> List[ItemRecommendResponse]:
    # 요청의 카테고리를 가져와서 유사한 아이템을 찾습니다.
    new_creator_data = {
        'category': request.channel_category,
        'subscriber_count': request.subscribers
    }

    # `recommend_for_new_creator` 함수에서 `h_item`과 `item_data` 사용
    recommended_item_ids = recommend_for_new_creator(new_creator_data, h_item, item_data, k=10)

    # 추천 결과를 ItemRecommendResponse 형식으로 반환
    return [
        ItemRecommendResponse(
            item_id=item_id,
            title=item_data[item_id]["title"],
            item_category=item_data[item_id]["category"],
            media_type=item_data[item_id]["media_type"],
            score=item_data[item_id]["score"]
        )
        for item_id in recommended_item_ids
    ]


# 새로운 아이템 요청을 처리하고, 유사한 크리에이터를 추천합니다.
def recommend_for_item(request: ItemRecommendRequest, creator_data) -> List[CreatorRecommendResponse]:
    new_item_data = {
        'title': request.title,
        'description': '',  # 추가 설명이 필요할 수 있음
        'category': request.item_category
    }

    recommended_creator_ids = recommend_for_new_item(new_item_data, model, h_item, k=10)

    # 추천 결과를 CreatorRecommendResponse 형식으로 반환
    return [
        CreatorRecommendResponse(
            user_id=creator_id,
            channel_name=creator_data[creator_id]["channel_name"],
            channel_category=creator_data[creator_id]["channel_category"],
            subscribers=creator_data[creator_id]["subscribers"]
        )
        for creator_id in recommended_creator_ids
    ]
