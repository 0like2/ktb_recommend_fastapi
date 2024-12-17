from src.rec_system.model_lightgcn import world
from src.rec_system.model_llm.model_llm4rec import TextEmbedder, load_data, generate_graph, LLMCandidateRanker
from src.rec_system.method.recommendation import Recommender
from src.rec_system.model_lightgcn.recommendation_light import LightGCNRecommender
from src.rec_system.model_lightgcn.dataloader import SimilarityDataset
from src.rec_system.model_llm.model_llm4rec import (
    recommend_for_new_creator,
    recommend_for_new_item
)
# joing-ai에서 수정 예정
import os
from dotenv import load_dotenv


class RecommendationService:
    def __init__(self):
        # NeuMF 모델 초기화
        self.neumf_recommender = Recommender(
            model_path="src/rec_system/model/output/neumf_factor8neg4_Epoch4_HR1.0000_NDCG1.0000.model",
            config_path="src/rec_system/model/output/config/config_epoch_4.pkl"
        )

        # LLM 모델 초기화
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Error: OPENAI_API_KEY not found. Please set the API key in your environment.")
        self.api_key = api_key
        self.embedder = TextEmbedder(api_key=self.api_key, model_name="text-embedding-3-small")
        creator_path_llm = "src/rec_system/model_lightgcn/input/Creator_random25.csv"
        item_path_llm = "src/rec_system/model_lightgcn/input/Item_random25.csv"
        self.creators_df, self.items_df = load_data(creator_path_llm, item_path_llm)
        self.connections = generate_graph(self.creators_df, self.items_df, self.embedder)
        self.llm_ranker = LLMCandidateRanker(api_key=self.api_key, model_name="gpt-4", temperature=0.5)

        # LightGCN 모델 초기화
        model_path_light = "src/rec_system/model_lightgcn/output/checkpoints/lgn-custom-similarity-3-64.pth.tar"
        creator_file_light = world.config['creator_file']
        item_file_light = world.config['item_file']
        similarity_matrix_file = world.config['similarity_matrix_file']
        self.dataset = SimilarityDataset(
            creator_file=creator_file_light,
            item_file=item_file_light,
            similarity_matrix_file=similarity_matrix_file,
            config=world.config
        )
        self.lightgcn_recommender = LightGCNRecommender(model_path_light, self.dataset)

        # 모델별 가중치 설정
        self.model_weights = {
            "neumf": 0.3,  # NeuMF의 가중치
            "lightgcn": 0.3,  # LightGCN의 가중치
            "llm": 0.4  # LLM의 가중치
        }

    def _ensure_unique_id_llm(self, data, is_item=True):
        if is_item:
            if "item_id" not in data:
                max_item_id = self.items_df['item_id'].max()
                data["item_id"] = max_item_id + 1
        else:
            if "creator_id" not in data:
                max_creator_id = self.creators_df['creator_id'].max()
                data["creator_id"] = max_creator_id + 1
        return data

    # NeuMF 추천 함수
    def recommend_for_new_item_neumf(self, item_data):
        return self.neumf_recommender.recommend_for_new_item(item_data)

    def recommend_for_new_creator_neumf(self, creator_data):
        return self.neumf_recommender.recommend_for_new_creator(creator_data)

    # LightGCN 추천 함수
    def recommend_for_new_item_lightgcn(self, item_data):
        processed_item = self.lightgcn_recommender.process_new_item(item_data)
        return self.lightgcn_recommender.recommend_for_new_item(processed_item)

    def recommend_for_new_creator_lightgcn(self, creator_data):
        processed_creator = self.lightgcn_recommender.process_new_creator(creator_data)
        return self.lightgcn_recommender.recommend_for_new_creator(processed_creator)

    # LLM 추천 함수
    def recommend_for_new_item_llm(self, item_data):
        item_data = self._ensure_unique_id_llm(item_data, is_item=True)
        return recommend_for_new_item(
            item_data,
            creators_df=self.creators_df,
            items_df=self.items_df,
            embedder=self.embedder,
            connections=self.connections,
            llm_ranker=self.llm_ranker,
            top_k=10
        )

    def recommend_for_new_creator_llm(self, creator_data):
        creator_data = self._ensure_unique_id_llm(creator_data, is_item=False)
        return recommend_for_new_creator(
            creator_data,
            creators_df=self.creators_df,
            items_df=self.items_df,
            embedder=self.embedder,
            connections=self.connections,
            llm_ranker=self.llm_ranker,
            top_k=10
        )

    # 앙상블 추천
    def recommend_for_new_item(self, item_data):
        neumf_results = self.recommend_for_new_item_neumf(item_data)
        print("neumf_results:", neumf_results)
        lightgcn_results = self.recommend_for_new_item_lightgcn(item_data)
        print("lightgcn_results:", lightgcn_results)
        llm_results = self.recommend_for_new_item_llm(item_data)
        print("llm_results:", llm_results)

        final_results = self._weighted_ensemble(
            {"neumf": neumf_results, "lightgcn": lightgcn_results, "llm": llm_results}
        )
        return final_results

    def recommend_for_new_creator(self, creator_data):
        neumf_results = self.recommend_for_new_creator_neumf(creator_data)
        print("neumf_results:", neumf_results)
        lightgcn_results = self.recommend_for_new_creator_lightgcn(creator_data)
        print("lightgcn_results:", lightgcn_results)
        llm_results = self.recommend_for_new_creator_llm(creator_data)
        print("llm_results:", llm_results)

        final_results = self._weighted_ensemble(
            {"neumf": neumf_results, "lightgcn": lightgcn_results, "llm": llm_results}
        )
        return final_results

    # 가중치 기반 앙상블 로직
    def _weighted_ensemble(self, model_results):
        aggregated_scores = {}
        num_results = max(len(results) for results in model_results.values())

        for model_name, results in model_results.items():
            weight = self.model_weights.get(model_name, 1.0)
            for rank, rec in enumerate(results, start=1):
                rec_id = rec.get('creator_id', rec.get('item_id'))

                rank_score = (num_results - rank + 1) * weight

                if rec_id not in aggregated_scores:
                    aggregated_scores[rec_id] = {
                        **rec,
                        "score": rank_score
                    }
                else:
                    aggregated_scores[rec_id]["score"] += rank_score

        sorted_results = sorted(
            aggregated_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        # 상위 10개 추천 반환
        return sorted_results[:10]
