from fastapi import APIRouter, HTTPException
from src.rec_system.schemas import ItemRecommendRequest, CreatorRecommendRequest, CreatorRecommendResponse, \
    ItemRecommendResponse
from src.rec_system.service import RecommendationService
from src.rec_system.model_lightgcn.world import initialize_world, config

router = APIRouter()
recommendation_service = None


@router.on_event("startup")
async def startup_event():
    global recommendation_service
    initialize_world()
    recommendation_service = RecommendationService()


@router.post("/rec/recommend/creator", response_model=CreatorRecommendResponse)
def recommend_item(data: ItemRecommendRequest):
    global recommendation_service
    try:
        recommendations = recommendation_service.recommend_for_new_item(data.dict())
        return {
            "recommended_creators": [
                {
                    "creator_id": rec["creator_id"],
                    "channel_category": rec["channel_category"],
                    "channel_name": rec["channel_name"],
                    "subscribers": (
                        int(rec["subscribers"].replace(",", "")) if isinstance(rec["subscribers"], str) else rec[
                            "subscribers"]
                    )
                }
                for rec in recommendations
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rec/recommend/item", response_model=ItemRecommendResponse)
def recommend_creator(data: CreatorRecommendRequest):
    global recommendation_service
    try:
        recommendations = recommendation_service.recommend_for_new_creator(data.dict())
        return {
            "recommended_items": [
                {
                    "item_id": rec["item_id"],
                    "title": rec["title"],
                    "item_category": rec["item_category"],
                    "media_type": rec["media_type"],
                    "score": int(rec["score"]),
                    "item_content": rec["item_content"]
                }
                for rec in recommendations
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
