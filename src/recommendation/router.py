from fastapi import APIRouter
from schemas import CreatorRecommendRequest, CreatorRecommendResponse, ItemRecommendRequest, \
    ItemRecommendResponse
from service import recommend_for_creator, recommend_for_item

router = APIRouter()


# Creator recommend Endpoint
@router.post("/ai/recommendation/creator")
def creator_recommendation_router(request: CreatorRecommendRequest):
    return recommend_for_creator(request)


# Item Recommend Endpoint
@router.post("/ai/recommendation/item")
def item_recommendation_router(request: ItemRecommendRequest):
    return recommend_for_item(request)
