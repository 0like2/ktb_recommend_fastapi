from pydantic import BaseModel
from typing import List
from pydantic import BaseModel, Field, field_validator
from fastapi import HTTPException

CATEGORY_MAPPING = {
    "GAME": "game",
    "TECH": "tech",
    "EDUCATION": "education",
    "KNOWHOW_STYLE": "beauty",
    "NEWS_POLITICS": "economy",
    "SPORTS": "sports_health",
    "NONPROFIT_SOCIAL": "government",
    "PETS_ANIMALS": "pet",
    "ENTERTAINMENT": "entertainment",
    "TRAVEL_EVENTS": "travel",
    "MOVIE_ANIMATION": "movie",
    "MUSIC": "music",
    "PEOPLE_BLOG": "celeb",
    "AUTO_TRANSPORT": "car_auto",
    "COMEDY": "entertainment",
    "ETC": "hobbies",
    "KIDS": "kids",
    "FOOD_COOKING": "food_cooking"
}


class CreatorRecommendRequest(BaseModel):
    channel_name: str
    channel_category: str
    subscribers: int

    @field_validator("channel_category")
    def map_channel_category(cls, value):
        if value not in CATEGORY_MAPPING:
            raise HTTPException(status_code=422, detail={
                "code": "INVALID_CATEGORY",
                "message": "유효하지 않은 크리에이터 카테고리입니다."
            })
        return CATEGORY_MAPPING[value]

    @field_validator("channel_name", "channel_category", "subscribers")
    def validate_creator(cls, value, info):
        field_name = info.field_name
        if (field_name in ["channel_name", "channel_category"] and not value.strip()) or (
                field_name == "subscribers" and value < 0):
            raise HTTPException(status_code=422,
                                detail={"code": 'INVALID_CREATOR_INPUT', "message": "유효하지 않은 크리에이터 입력값입니다."})
        return value


class ItemRecommendRequest(BaseModel):
    title: str
    item_category: str
    media_type: str
    score: float
    item_content: str

    @field_validator("item_category")
    def map_item_category(cls, value):
        if value not in CATEGORY_MAPPING:
            raise HTTPException(status_code=422, detail={
                "code": "INVALID_CATEGORY",
                "message": "유효하지 않은 아이템 카테고리입니다."
            })
        return CATEGORY_MAPPING[value]

    @field_validator("title", "item_category", "media_type", "score", "item_content")
    def validate_item(cls, value, info):
        field_name = info.field_name
        if (field_name in ["title", "item_category", "item_category"] and not value.strip()) or (
                field_name == "score" and value < 0) or (
                field_name == "media_type" and value.lower() not in ["short_form", "long_form"]):
            raise HTTPException(status_code=422,
                                detail={"code": 'INVALID_ITEM_INPUT', "message": "유효하지 않은 아이템 입력값입니다."})
        return value


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
    score: float
    item_content: str


# item -> creator 추천
class CreatorRecommendResponse(BaseModel):
    recommended_creators: List[RecommendCreator]


# creator -> item 추천
class ItemRecommendResponse(BaseModel):
    recommended_items: List[RecommendItem]
