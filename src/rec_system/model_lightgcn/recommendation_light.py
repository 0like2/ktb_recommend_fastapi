import torch
import warnings
from src.rec_system.model_lightgcn import world
from src.rec_system.model_lightgcn.model import LightGCN
from src.rec_system.model_lightgcn.dataloader import SimilarityDataset

warnings.filterwarnings("ignore", category=FutureWarning)


class LightGCNRecommender:
    def __init__(self, model_path, dataset):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load LightGCN model
        self.model = LightGCN(dataset.config, dataset).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Load metadata
        self.dataset = dataset
        self.user_embeddings, self.item_embeddings = self.model.get_embeddings()

        # Linear layer for embedding size reduction
        embedding_dim = self.user_embeddings.shape[1]
        bert_dim = 384
        self.embedding_reducer = torch.nn.Linear(bert_dim, embedding_dim).to(self.device)
        self.embedding_reducer.eval()

        self.item_category_mapping = dataset.item_category_mapping
        self.channel_category_mapping = dataset.channel_category_mapping

    def process_new_item(self, item_data):
        """
        Preprocess new item data for recommendation.
        """
        item_category = item_data.get('item_category', 'unknown')

        # 매핑된 카테고리 인덱스 가져오기
        category_index = list(self.item_category_mapping.values()).index(item_category)
        media_type = 0 if item_data.get('media_type', '').lower() == 'short' else 1

        # 아이템 임베딩 처리
        item_embedding = torch.tensor(
            self.dataset.text_embedder.get_text_embedding(item_data.get('title', 'unknown')),
            dtype=torch.float
        ).to(self.device)
        reduced_item_embedding = self.embedding_reducer(item_embedding)

        processed_item = {
            'item_id': self.dataset.m_items,
            'item_category': category_index,
            'media_type': media_type,
            'item_embedding': reduced_item_embedding,
        }
        return processed_item

    def process_new_creator(self, creator_data):
        """
        Preprocess new creator data for recommendation.
        """
        channel_category = creator_data.get('channel_category', 'unknown')

        # 매핑된 카테고리 인덱스 가져오기
        category_index = list(self.channel_category_mapping.values()).index(channel_category)
        # max_value와 scale을 전달하여 구독자 수를 정규화
        normalized_subscribers = self.dataset.normalize_subscribers(
            creator_data.get('subscribers', 0),
            max_value=10000000,  # 데이터셋 내 최대 구독자 수
            scale=100
        )

        # 크리에이터 임베딩 처리
        creator_embedding = torch.tensor(
            self.dataset.text_embedder.get_text_embedding(creator_data.get('channel_name', 'unknown')),
            dtype=torch.float
        ).to(self.device)

        # 384차원의 임베딩을 64차원으로 축소
        reduced_creator_embedding = self.embedding_reducer(creator_embedding)

        processed_creator = {
            'creator_id': self.dataset.n_users,
            'channel_category': category_index,
            'creator_embedding': reduced_creator_embedding,
            'subscribers': normalized_subscribers,
        }
        return processed_creator

    def recommend_for_new_item(self, processed_item, top_k=10):
        """
        Recommends users for a new item.
        """
        item_embedding = processed_item['item_embedding']

        # Compute similarity between the new item and all user embeddings
        scores = torch.matmul(self.user_embeddings, item_embedding)
        top_k_indices = torch.topk(scores, top_k).indices.cpu().numpy()

        # Retrieve recommended user metadata
        recommended_users = [
            {
                'creator_id': int(i)+1,
                'channel_name': self.dataset.creators.iloc[i]['channel_name'],
                'channel_category': self.dataset.channel_category_mapping[
                    self.dataset.creators.iloc[i]['channel_category']],
                'subscribers': int(self.dataset.creators.iloc[i]['subscribers']),
            }
            for i in top_k_indices
        ]
        return recommended_users

    def recommend_for_new_creator(self, processed_creator, top_k=10):
        """
        Recommends items for a new creator.
        """
        creator_embedding = processed_creator['creator_embedding']

        # Compute similarity between the new creator and all item embeddings
        scores = torch.matmul(self.item_embeddings, creator_embedding)
        top_k_indices = torch.topk(scores, top_k).indices.cpu().numpy()

        # Retrieve recommended item metadata
        recommended_items = [
            {
                'item_id': int(i)+1,
                'title': self.dataset.items.iloc[i]['title'],
                'item_category': self.dataset.item_category_mapping[self.dataset.items.iloc[i]['item_category']],
                'media_type': self.dataset.media_type_mapping[self.dataset.items.iloc[i]['media_type']],
                'item_score': int(self.dataset.items.iloc[i]['score']),
                'item_content': self.dataset.items.iloc[i]['item_content'],
            }
            for i in top_k_indices
        ]
        return recommended_items


if __name__ == "__main__":
    # Use paths from world.py
    model_path = "output/checkpoints/lgn-custom-similarity-3-64.pth.tar"
    creator_file = world.config['creator_file']
    item_file = world.config['item_file']
    similarity_matrix_file = world.config['similarity_matrix_file']

    # Initialize dataset and recommender
    dataset = SimilarityDataset(
        creator_file=creator_file,
        item_file=item_file,
        similarity_matrix_file=similarity_matrix_file,
        config=world.config
    )
    print("Dataset loaded successfully.")

    # Initialize recommender
    recommender = LightGCNRecommender(model_path, dataset)
    print("Model loaded successfully.")

    # New item example
    new_item_data = {
        'title': "바밤바를 뛰어넘는 밤 맛 과자가 있을까?",
        'item_category': "entertainment",
        'media_type': 'short-form',
        'score': 80,
        'item_content': '다양한 밤 맛 과자를 비교하며 맛과 질감을 리뷰하는 콘텐츠'
    }
    processed_item = recommender.process_new_item(new_item_data)
    recommended_users = recommender.recommend_for_new_item(processed_item)
    print(f"추천 사용자 목록: {recommended_users}")

    # New creator example
    new_creator_data = {
        'channel_category': "tech",
        'channel_name': "최마태의 POST IT",
        'subscribers': 263000
    }
    processed_creator = recommender.process_new_creator(new_creator_data)
    recommended_items = recommender.recommend_for_new_creator(processed_creator)
    print(f"추천 아이템 목록: {recommended_items}")
