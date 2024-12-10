from rec_system.model_lightgcn import world
from src.rec_system.method.recommendation import Recommender
from src.rec_system.model_lightgcn.recommendation_light import LightGCNRecommender
from src.rec_system.model_lightgcn.dataloader import SimilarityDataset
import src.rec_system.model_lightgcn.world


class RecommendationService:
    def __init__(self):
        # NeuMF 모델 초기화
        self.neumf_recommender = Recommender(
            model_path="src/rec_system/model/output/neumf_model_path",
            config_path="src/rec_system/model/output/config_neumf.pkl"
        )

        # LLM 모델 초기화
        self.llm_recommender = Recommender(
            model_path="src/rec_system/model_llm/output/llm_model_path",
            config_path="src/rec_system/model_llm/output/config_llm.pkl"
        )

        # LightGCN 모델 초기화
        model_path_light = "output/checkpoints/lgn-custom-similarity-3-64.pth.tar"
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
            "neumf": 0.3,       # NeuMF의 가중치
            "lightgcn": 0.3,    # LightGCN의 가중치
            "llm": 0.4          # LLM의 가중치
        }

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
        return self.llm_recommender.recommend_for_new_item(item_data)

    def recommend_for_new_creator_llm(self, creator_data):
        return self.llm_recommender.recommend_for_new_creator(creator_data)

    # 앙상블 추천
    def recommend_for_new_item(self, item_data):
        neumf_results = self.recommend_for_new_item_neumf(item_data)
        lightgcn_results = self.recommend_for_new_item_lightgcn(item_data)
        llm_results = self.recommend_for_new_item_llm(item_data)

        final_results = self._weighted_ensemble(
            {"neumf": neumf_results, "lightgcn": lightgcn_results, "llm": llm_results}
        )
        return final_results

    def recommend_for_new_creator(self, creator_data):
        neumf_results = self.recommend_for_new_creator_neumf(creator_data)
        lightgcn_results = self.recommend_for_new_creator_lightgcn(creator_data)
        llm_results = self.recommend_for_new_creator_llm(creator_data)

        final_results = self._weighted_ensemble(
            {"neumf": neumf_results, "lightgcn": lightgcn_results, "llm": llm_results}
        )
        return final_results

    # 가중치 기반 앙상블 로직
    def _weighted_ensemble(self, model_results):
        """
        모델별 추천 결과를 가중치 기반으로 앙상블.
        """
        aggregated_results = {}

        # 모델별 결과와 가중치 처리
        for model_name, results in model_results.items():
            weight = self.model_weights.get(model_name, 1.0)  # 가중치 기본값은 1.0
            for rec in results:
                rec_id = rec['creator_id'] if 'creator_id' in rec else rec['item_id']
                if rec_id not in aggregated_results:
                    aggregated_results[rec_id] = {
                        **rec,
                        "score": weight  # 초기 점수는 가중치로 설정
                    }
                else:
                    aggregated_results[rec_id]["score"] += weight  # 가중치 누적

        # 점수를 기준으로 정렬
        sorted_results = sorted(
            aggregated_results.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        # 상위 10개 추천 반환
        return sorted_results[:10]
