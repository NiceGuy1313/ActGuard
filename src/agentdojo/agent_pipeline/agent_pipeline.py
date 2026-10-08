import importlib.resources
import json
import logging
import os
from collections.abc import Iterable, Sequence
from functools import partial
from typing import Literal

import openai
import requests
import yaml
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
from agentdojo.agent_pipeline.reflective_audit import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LOGPROB_DELTA_THRESHOLD,
    DEFAULT_LOGPROB_MODEL,
    ReflectiveAuditGuard,
    ReflectiveAuditTextProtocolLLM,
)
from agentdojo.agent_pipeline.tool_execution import (
    ToolsExecutionLoop,
    ToolsExecutor,
    tool_result_to_str,
)
from agentdojo.functions_runtime import EmptyEnv, Env, FunctionsRuntime
from agentdojo.logging import Logger
from agentdojo.models import MODEL_PROVIDERS, ModelsEnum
from agentdojo.openai_client_config import openai_client_kwargs
from agentdojo.types import ChatMessage

DEFENSES = [
    "reflective_audit",
]
"""Defenses shipped by the ActGuard codebase."""


def load_system_message(system_message_name: str | None) -> str:
    package_files = importlib.resources.files("agentdojo")
    path = package_files / "data" / "system_messages.yaml"
    with importlib.resources.as_file(path) as p, p.open() as f:
        system_messages = yaml.safe_load(f)
    return system_messages.get(system_message_name, system_messages["default"])


def _get_local_model_id(port) -> str:
    url = f"http://localhost:{port}/v1/models"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    model_name = data["data"][0]["id"]
    logging.info(f"Using model: {model_name}")
    return model_name


def get_llm(provider: str, model: str, model_id: str | None, tool_delimiter: str) -> BasePipelineElement:
    if provider == "openai":
        client = openai.OpenAI(**openai_client_kwargs())
        llm = OpenAILLM(client, model)
    elif provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set when using a direct Claude model")
        client_kwargs = openai_client_kwargs(
            api_key=api_key,
            base_url="https://api.anthropic.com/v1/",
        )
        workspace_id = os.getenv("ANTHROPIC_WORKSPACE_ID")
        if workspace_id:
            client_kwargs["default_headers"] = {"anthropic-workspace-id": workspace_id}
        client = openai.OpenAI(**client_kwargs)
        llm = OpenAILLM(client, model)
    elif provider == "openrouter":
        client = openai.OpenAI(
            **openai_client_kwargs(
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            )
        )
        llm = OpenAILLM(client, model)
    elif provider == "mulerouter":
        client = openai.OpenAI(
            **openai_client_kwargs(
                api_key=os.getenv("MULEROUTER_API_KEY"),
                base_url="https://api.mulerouter.ai/vendors/openai/v1",
            )
        )
        llm = OpenAILLM(client, model)
    elif provider == "together":
        client = openai.OpenAI(
            **openai_client_kwargs(
                api_key=os.getenv("TOGETHER_API_KEY"),
                base_url="https://api.together.xyz/v1",
            )
        )
        llm = OpenAILLM(client, model)
    elif provider == "vllm_parsed":
        port = os.getenv("LOCAL_LLM_PORT", 8000)
        client = openai.OpenAI(
            **openai_client_kwargs(
                api_key="EMPTY",
                base_url=f"http://localhost:{port}/v1",
            )
        )
        llm = OpenAILLM(client, _get_local_model_id(port))
    else:
        raise ValueError(f"Unsupported ActGuard provider: {provider}")
    return llm


def _resolve_model_identifier(model: str) -> str:
    """Resolve a ModelsEnum member name or value to the backend model id."""
    if model in ModelsEnum.__members__:
        return str(ModelsEnum[model])
    try:
        return str(ModelsEnum(model))
    except ValueError:
        return model


class PipelineConfig(BaseModel):
    # to allow union type between str and BasePipelineElement
    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: str | BasePipelineElement
    """Which LLM to use. One of the models in [`ModelsEnum`][agentdojo.models.ModelsEnum]
    or a custom object which inherits from [`BasePipelineElement`][agentdojo.agent_pipeline.base_pipeline_element.BasePipelineElement]
    and implements calls to an LLM."""
    model_id: str | None
    """LLM model id for local models."""
    defense: str | None
    """Which defense to use. One of the defenses in [`DEFENSES`][agentdojo.agent_pipeline.agent_pipeline.DEFENSES]."""
    tool_delimiter: str = "tool"
    """Which tool delimiter to use."""
    system_message_name: str | None
    """The name of the system message to use. If not provided, the default system message will be used."""
    system_message: str | None
    """The system message to use. If not provided, the default system message will be used. If provided, it will
    override `system_message_name`."""
    tool_output_format: Literal["yaml", "json"] | None = None
    """Format to use for tool outputs. If not provided, the default format is yaml."""
    suite_name: str | None = None
    """Suite name, used by external defense adapters when needed."""
    reflective_audit_logprob_model: str | None = None
    """Local causal LM path/name used for reflective_audit contrastive logprob scoring."""
    reflective_audit_verifier_model: str | None = None
    """OpenAI-compatible model name used to judge whether localized chunks contain IPI."""
    reflective_audit_embedding_model: str | None = None
    """Embedding model path/name for the reflective_audit defense."""
    reflective_audit_top_k: int | None = None
    """Number of localized chunks retrieved before reflective_audit attribution/verifier expansion."""
    reflective_audit_logprob_delta_threshold: float | None = None
    """Minimum contrastive logprob delta required before reflective_audit sends chunks to the verifier."""

    @model_validator(mode="after")
    def validate_system_message(self) -> Self:
        if self.system_message is not None:
            return self
        self.system_message = load_system_message(self.system_message_name)
        return self


class AgentPipeline(BasePipelineElement):
    """Executes a sequence of [`BasePipelineElement`][agentdojo.agent_pipeline.BasePipelineElement]s in order.

    Args:
        elements: the elements of the pipeline to execute.
    """

    def __init__(self, elements: Iterable[BasePipelineElement]) -> None:
        self.elements = elements

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict | None = None,
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        if extra_args is None:
            extra_args = {}
        logger = Logger().get()
        for element in self.elements:
            query, runtime, env, messages, extra_args = element.query(query, runtime, env, messages, extra_args)
            logger.log(messages)
        return query, runtime, env, messages, extra_args

    @classmethod
    def from_config(cls, config: PipelineConfig) -> Self:
        """Creates a pipeline for a given model and defense."""
        # TODO: make this more elegant
        llm = (
            get_llm(MODEL_PROVIDERS[ModelsEnum(config.llm)], config.llm, config.model_id, config.tool_delimiter)
            if isinstance(config.llm, str)
            else config.llm
        )
        llm_name = config.llm if isinstance(config.llm, str) else llm.name

        assert config.system_message is not None
        system_message_component = SystemMessage(config.system_message)
        init_query_component = InitQuery()

        if config.tool_output_format == "json":
            tool_output_formatter = partial(tool_result_to_str, dump_fn=json.dumps)
        else:
            tool_output_formatter = tool_result_to_str

        if config.defense is None:
            tools_loop = ToolsExecutionLoop([ToolsExecutor(tool_output_formatter), llm])
            pipeline = cls([system_message_component, init_query_component, llm, tools_loop])
            pipeline.name = llm_name
            return pipeline
        if config.defense == "reflective_audit":
            if not isinstance(llm, OpenAILLM):
                raise ValueError("Reflective audit is only supported for OpenAI-compatible models")
            if not isinstance(llm_name, str):
                raise ValueError("Reflective audit requires a named model")
            reflective_audit_logprob_model = (
                config.reflective_audit_logprob_model
                or os.getenv("REFLECTIVE_AUDIT_LOGPROB_MODEL", DEFAULT_LOGPROB_MODEL)
            )
            reflective_audit_verifier_model = (
                config.reflective_audit_verifier_model
                or os.getenv("REFLECTIVE_AUDIT_VERIFIER_MODEL", llm_name)
            )
            reflective_audit_verifier_model = _resolve_model_identifier(reflective_audit_verifier_model)
            reflective_audit_embedding_model = (
                config.reflective_audit_embedding_model
                or os.getenv("REFLECTIVE_AUDIT_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
            )
            reflective_audit_top_k = (
                config.reflective_audit_top_k
                if config.reflective_audit_top_k is not None
                else int(os.getenv("REFLECTIVE_AUDIT_TOP_K", "3"))
            )
            reflective_audit_logprob_delta_threshold = (
                config.reflective_audit_logprob_delta_threshold
                if config.reflective_audit_logprob_delta_threshold is not None
                else float(
                    os.getenv(
                        "REFLECTIVE_AUDIT_LOGPROB_DELTA_THRESHOLD",
                        str(DEFAULT_LOGPROB_DELTA_THRESHOLD),
                    )
                )
            )
            reflective_llm = ReflectiveAuditTextProtocolLLM(
                llm.client,
                llm_name,
                reasoning_effort=llm.reasoning_effort,
                temperature=llm.temperature,
            )
            tools_loop = ToolsExecutionLoop(
                [
                    ReflectiveAuditGuard(
                        llm.client,
                        llm_name,
                        logprob_model=reflective_audit_logprob_model,
                        verifier_model=reflective_audit_verifier_model,
                        embedding_model=reflective_audit_embedding_model,
                        top_k=reflective_audit_top_k,
                        logprob_delta_threshold=reflective_audit_logprob_delta_threshold,
                    ),
                    ToolsExecutor(tool_output_formatter),
                    reflective_llm,
                ]
            )
            pipeline = cls([system_message_component, init_query_component, reflective_llm, tools_loop])
            pipeline.name = f"{llm_name}-{config.defense}"
            pipeline.run_config_name = "actguard"
            return pipeline
        raise ValueError("Invalid defense name")
