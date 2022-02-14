from adaptive_xgboost import AdaptiveXGBoostClassifier
from adaptive_semiV2 import AdaptiveFS
from adaptive_semi import AdaptiveSemi

from skmultiflow.data import ConceptDriftStream
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.data.file_stream import FileStream
from skmultiflow.data.random_tree_generator import RandomTreeGenerator
from skmultiflow.data import SEAGenerator
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.data.hyper_plane_generator  import HyperplaneGenerator

# Adaptive XGBoost classifier parameters
n_estimators = 30       # Number of members in the ensemble
learning_rate = 0.01     # Learning rate or eta
max_depth = 3        # Max depth for each tree in the ensemble
max_window_size = 4096  # Max window size
min_window_size = 1     # set to activate the dynamic window strategy
detect_drift = True    # Enable/disable drift detection
ratio_unsampled = 0
small_window_size = 150

max_buffer = 50
pre_train = 20


AXGBsemi = AdaptiveSemi(learning_rate=learning_rate,
                                  max_depth=max_depth,
                                  max_window_size=max_window_size,
                                  min_window_size=min_window_size,
                                  ratio_unsampled=ratio_unsampled,
                                  small_window_size=small_window_size,
                                  max_buffer=max_buffer,
                                  pre_train=pre_train)

AXGBr = AdaptiveXGBoostClassifier(update_strategy='replace',
                                  n_estimators=n_estimators,
                                  learning_rate=learning_rate,
                                  max_depth=max_depth,
                                  max_window_size=max_window_size,
                                  min_window_size=min_window_size,
                                  detect_drift=detect_drift,
                                  ratio_unsampled=ratio_unsampled)


AXGBfs = AdaptiveFS(learning_rate=learning_rate,
                                  max_depth=max_depth,
                                  max_window_size=max_window_size,
                                  min_window_size=min_window_size,
                                  ratio_unsampled=ratio_unsampled,
                                  small_window_size=small_window_size,
                                  max_buffer=max_buffer,
                                  pre_train=pre_train)

# ## meu thread
# AXGBt = Adaptive3(n_estimators=n_estimators,
#                                   learning_rate=learning_rate,
#                                   max_depth=max_depth,
#                                   max_window_size=max_window_size,
#                                   min_window_size=min_window_size,
#                                   detect_drift=detect_drift)

# stream = FileStream("Crop_mapping.csv")
# print("aqui : ", stream.n_features)
# stream = SEAGenerator(classification_function = 0, random_state = 456, balance_classes = True, noise_percentage=0.28)
stream = ConceptDriftStream(random_state=1, position=25000)

# stream = RandomTreeGenerator(tree_random_state=8873, sample_random_state=69, n_classes=2, n_cat_features=600,
#                                  n_num_features=1000, n_categories_per_cat_feature=2, max_tree_depth=9, min_leaf_depth=3,
#                                  fraction_leaves_per_level=0.15)
# stream.prepare_for_use()   # Required for skmultiflow v0.4.1
print("num features: ", stream.n_features)



evaluator = EvaluatePrequential(pretrain_size=0,
                                max_samples=50000,
                                # batch_size=2048,
                                show_plot=True,
                                metrics=["accuracy","running_time"])

evaluator.evaluate(stream=stream,
                   model=[AXGBsemi],
                   model_names=["AXGB a"])
