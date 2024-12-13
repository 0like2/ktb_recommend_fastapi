from pydantic import BaseModel
from typing import List
from pydantic import Basemodel, Field, field_validator
from fastapi import HTTPException

class CreatorRecommendRequest(BaseModel):
    channel_name: str
    channel_category: str
    subscribers: int


    @field_validator("channel_name", "channel_category", "subscribers")
    def validate_creator(cls, value, field):
        if (field.name in ["channel_name", "channel_category"] and not value.strip()) or (field.name == "subscribers" and value < 0):
            raise HTTPException(status_code=422, detail="유효하지 않은 크리에이터 입력값입니다.")
        return value

class ItemRecommendRequest(BaseModel):
    title: str
    item_category: str
    media_type: str
    score: int
    item_content: str


class RecommendCreator(BaseModel):
    creator_id: int
    channel_category: str
    channel_name: str
    subscribers: int


class RecommendItem(BaseModel):
    item_id: int
    title: str
    item_category: str
    media_type: str
    score: int
    item_content: str

# item -> creator 추천
class CreatorRecommendResponse(BaseModel):
    recommended_creators: List[RecommendCreator]


# creator -> item 추천
class ItemRecommendResponse(BaseModel):
    recommended_items: List[RecommendItem]
