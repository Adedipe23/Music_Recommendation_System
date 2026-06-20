from src.data_loader import load_interactions
from src.recommender import HybridMusicRecommender


def main() -> None:
    interactions, source = load_interactions()
    model = HybridMusicRecommender().fit(interactions)
    model.save_model()
    print(f"Dataset: {source['label']}")
    print(model.get_dataset_summary())
    print("\nRecommendations for Tems:")
    print(model.recommend_artists("Tems", 5).to_string(index=False))


if __name__ == "__main__":
    main()
