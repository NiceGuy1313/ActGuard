from agentdojo.strenum import StrEnum


class ModelsEnum(StrEnum):
    """OpenAI-compatible models supported by the final ActGuard release."""

    GPT_5_1_2025_11_13 = "gpt-5.1-2025-11-13"
    GPT_5_MINI_2025_08_07 = "gpt-5-mini-2025-08-07"
    GPT_4O_2024_08_06 = "gpt-4o-2024-08-06"
    GPT_4O_2024_05_13 = "gpt-4o-2024-05-13"
    GPT_4O_MINI_2024_07_18 = "gpt-4o-mini-2024-07-18"
    GPT_4_TURBO_2024_04_09 = "gpt-4-turbo-2024-04-09"
    GPT_4_0125_PREVIEW = "gpt-4-0125-preview"
    GPT_3_5_TURBO_0125 = "gpt-3.5-turbo-0125"

    OPENROUTER_QWEN_2_5_72B = "qwen/qwen-2.5-72b-instruct"
    OPENROUTER_QWEN3_235B = "qwen/qwen3-235b-a22b-2507"
    OPENROUTER_LLAMA_3_3_70B = "meta-llama/llama-3.3-70b-instruct:free"
    OPENROUTER_GEMINI_2_5_PRO = "google/gemini-2.5-pro"
    OPENROUTER_GEMINI_2_5_FLASH = "google/gemini-2.5-flash"
    OPENROUTER_GEMINI_3_PRO_PREVIEW = "google/gemini-3-pro-preview"
    OPENROUTER_GEMINI_3_FLASH_PREVIEW = "google/gemini-3-flash-preview"
    OPENROUTER_GEMINI_3_1_FLASH_LITE = "google/gemini-3.1-flash-lite"
    OPENROUTER_GPT_4O_MINI_2024_07_18 = "openai/gpt-4o-mini-2024-07-18"
    OPENROUTER_GPT_5_MINI = "openai/gpt-5-mini"
    OPENROUTER_DEEPSEEK_V4_FLASH = "deepseek/deepseek-v4-flash"

    QWEN3_MAX = "qwen3-max"
    VLLM_PARSED = "vllm_parsed"


MODEL_PROVIDERS = {
    ModelsEnum.GPT_5_1_2025_11_13: "openai",
    ModelsEnum.GPT_5_MINI_2025_08_07: "openai",
    ModelsEnum.GPT_4O_2024_08_06: "openai",
    ModelsEnum.GPT_4O_2024_05_13: "openai",
    ModelsEnum.GPT_4O_MINI_2024_07_18: "openai",
    ModelsEnum.GPT_4_TURBO_2024_04_09: "openai",
    ModelsEnum.GPT_4_0125_PREVIEW: "openai",
    ModelsEnum.GPT_3_5_TURBO_0125: "openai",
    ModelsEnum.OPENROUTER_QWEN_2_5_72B: "openrouter",
    ModelsEnum.OPENROUTER_QWEN3_235B: "openrouter",
    ModelsEnum.OPENROUTER_LLAMA_3_3_70B: "openrouter",
    ModelsEnum.OPENROUTER_GEMINI_2_5_PRO: "openrouter",
    ModelsEnum.OPENROUTER_GEMINI_2_5_FLASH: "openrouter",
    ModelsEnum.OPENROUTER_GEMINI_3_PRO_PREVIEW: "openrouter",
    ModelsEnum.OPENROUTER_GEMINI_3_FLASH_PREVIEW: "openrouter",
    ModelsEnum.OPENROUTER_GEMINI_3_1_FLASH_LITE: "openrouter",
    ModelsEnum.OPENROUTER_GPT_4O_MINI_2024_07_18: "openrouter",
    ModelsEnum.OPENROUTER_GPT_5_MINI: "openrouter",
    ModelsEnum.OPENROUTER_DEEPSEEK_V4_FLASH: "openrouter",
    ModelsEnum.QWEN3_MAX: "mulerouter",
    ModelsEnum.VLLM_PARSED: "vllm_parsed",
}


MODEL_NAMES = {
    "gpt-5.1-2025-11-13": "GPT-5",
    "gpt-5-mini-2025-08-07": "GPT-5 Mini",
    "gpt-4o-2024-08-06": "GPT-4o",
    "gpt-4o-2024-05-13": "GPT-4o",
    "gpt-4o-mini-2024-07-18": "GPT-4o Mini",
    "gpt-4-turbo-2024-04-09": "GPT-4 Turbo",
    "gpt-4-0125-preview": "GPT-4 Turbo",
    "gpt-3.5-turbo-0125": "GPT-3.5",
    "qwen/qwen-2.5-72b-instruct": "Qwen 2.5",
    "qwen/qwen3-235b-a22b-2507": "Qwen 3",
    "meta-llama/llama-3.3-70b-instruct:free": "Llama 3.3",
    "google/gemini-2.5-pro": "Gemini 2.5 Pro",
    "google/gemini-2.5-flash": "Gemini 2.5 Flash",
    "google/gemini-3-pro-preview": "Gemini 3 Pro",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "google/gemini-3.1-flash-lite": "Gemini 3.1 Flash Lite",
    "openai/gpt-4o-mini-2024-07-18": "GPT-4o Mini",
    "openai/gpt-5-mini": "GPT-5 Mini",
    "deepseek/deepseek-v4-flash": "DeepSeek V4 Flash",
    "qwen3-max": "Qwen 3 Max",
    "vllm_parsed": "Local OpenAI-compatible model",
}
