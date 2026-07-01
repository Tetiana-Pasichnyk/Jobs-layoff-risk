import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

from src.ml_model.DecisionTree_Predictor import train_model


def visualize_tree(pipeline):
    model = pipeline.named_steps['model']

    if 'preprocess' in pipeline.named_steps:
        preprocessor = pipeline.named_steps['preprocess']
        cat_transformer, sub_cat = preprocessor.transformers_[0][1], preprocessor.transformers_[0][2]
        sub_num = preprocessor.transformers_[1][2]
        cat_feature_names = cat_transformer.get_feature_names_out(sub_cat)
        feature_names = list(cat_feature_names) + list(sub_num)
    else:
        feature_names = None

    plt.figure(figsize=(22, 10))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=[str(c) for c in model.classes_],
        filled=True,
        rounded=True,
        fontsize=9
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    pipeline = train_model()
    visualize_tree(pipeline)
