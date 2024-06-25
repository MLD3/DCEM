BASE_PATH = "/data4/username/disparate_censorship_mitigation"
MASTER_DIR = "sweep_20230620_v2_all"
EXPERIMENTAL_PARAMS = ["target_prevalence", "testing_disparity", "prevalence_disparity", "k"]
MODEL_DIR_IDX = -1
EXPERIMENT_SWEEP_DIR_IDX = -3

def get_name_from_dir(dirname, idx):
    return dirname.split("/")[idx]

def get_model_name_from_dir(dirname):
    return get_name_from_dir(dirname, MODEL_DIR_IDX).split("_")[0]

def get_sweep_name_from_dir(dirname):
    return get_name_from_dir(dirname, EXPERIMENT_SWEEP_DIR_IDX)
