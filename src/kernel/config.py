###! Configuration
class config:
    ## Model directory
    MODEL = "MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF"
    ## Model quantization
    MODEL_QUANT = "*q6_K.gguf"
    ## Context size
    N_CTX = 2**12 - 1
    ## Offload layers
    N_GPU_LAYERS = 31
    ## Batch size
    N_BATCH = 512
    ## Temperature
    N_TEMPERATURE = 0.3
    ## Top K
    N_TOP_K = 40
    ## Top P
    N_TOP_P = 0.95
    ## Min P
    N_MIN_P = 0.05
    ## Commands Modules
    MODULES = []
    ## Initial task
    INITIAL_TASK = "No task..."
    ## Project directory
    PROJECT_DIR = ""
