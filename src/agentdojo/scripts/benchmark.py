import importlib
import logging
import warnings
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path
from typing import Literal

import click
from dotenv import load_dotenv
from rich import print
from rich.live import Live
from rich.logging import RichHandler

from agentdojo.agent_pipeline.agent_pipeline import DEFENSES, AgentPipeline, PipelineConfig
from agentdojo.agent_pipeline.reflective_audit import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LOGPROB_DELTA_THRESHOLD,
    DEFAULT_LOGPROB_MODEL,
)
from agentdojo.attacks.attack_registry import ATTACKS, load_attack
from agentdojo.benchmark import SuiteResults, benchmark_suite_with_injections, benchmark_suite_without_injections
from agentdojo.logging import OutputLogger
from agentdojo.models import ModelsEnum
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.task_suite.task_suite import TaskSuite


def benchmark_suite(
    suite: TaskSuite,
    model: ModelsEnum,
    logdir: Path,
    force_rerun: bool,
    benchmark_version: str,
    user_tasks: tuple[str, ...] = (),
    injection_tasks: tuple[str, ...] = (),
    model_id: str | None = None,
    attack: str | None = None,
    defense: str | None = None,
    tool_delimiter: str = "tool",
    system_message_name: str | None = None,
    system_message: str | None = None,
    live: Live | None = None,
    tool_output_format: Literal["yaml", "json"] | None = None,
    reflective_audit_logprob_model: str | None = None,
    reflective_audit_verifier_model: str | None = None,
    reflective_audit_embedding_model: str | None = None,
    reflective_audit_top_k: int | None = None,
    reflective_audit_logprob_delta_threshold: float | None = None,
) -> SuiteResults:
    if not load_dotenv(".env"):
        warnings.warn("No .env file found")

    print(f"Running benchmark for suite: '{suite.name}'")
    print(f"Using model: '{model}'")
    if attack is not None:
        print(f"Using attack: '{attack}'")
    if defense is not None:
        print(f"Using defense: '{defense}'")
    if len(user_tasks) > 0:
        print(f"Using user tasks: {', '.join(user_tasks)}")

    pipeline = AgentPipeline.from_config(
        PipelineConfig(
            llm=model,
            model_id=model_id,
            defense=defense,
            tool_delimiter=tool_delimiter,
            system_message_name=system_message_name,
            system_message=system_message,
            tool_output_format=tool_output_format,
            suite_name=suite.name,
            reflective_audit_logprob_model=reflective_audit_logprob_model,
            reflective_audit_verifier_model=reflective_audit_verifier_model,
            reflective_audit_embedding_model=reflective_audit_embedding_model,
            reflective_audit_top_k=reflective_audit_top_k,
            reflective_audit_logprob_delta_threshold=reflective_audit_logprob_delta_threshold,
        )
    )
    with OutputLogger(str(logdir), live=live):
        if attack is None:
            results = benchmark_suite_without_injections(
                pipeline,
                suite,
                user_tasks=user_tasks if len(user_tasks) != 0 else None,
                logdir=logdir,
                force_rerun=force_rerun,
                benchmark_version=benchmark_version,
            )
        else:
            attacker_ = load_attack(attack, suite, pipeline)
            results = benchmark_suite_with_injections(
                pipeline,
                suite,
                attacker_,
                user_tasks=user_tasks if len(user_tasks) != 0 else None,
                injection_tasks=injection_tasks if len(injection_tasks) != 0 else None,
                logdir=logdir,
                force_rerun=force_rerun,
                benchmark_version=benchmark_version,
            )
    print(f"Finished benchmark for suite: '{suite.name}'")

    return results


def benchmark_suite_from_name(
    suite_name: str,
    model: ModelsEnum,
    logdir: Path,
    force_rerun: bool,
    benchmark_version: str,
    user_tasks: tuple[str, ...] = (),
    injection_tasks: tuple[str, ...] = (),
    model_id: str | None = None,
    attack: str | None = None,
    defense: str | None = None,
    tool_delimiter: str = "tool",
    system_message_name: str | None = None,
    system_message: str | None = None,
    live: Live | None = None,
    tool_output_format: Literal["yaml", "json"] | None = None,
    reflective_audit_logprob_model: str | None = None,
    reflective_audit_verifier_model: str | None = None,
    reflective_audit_embedding_model: str | None = None,
    reflective_audit_top_k: int | None = None,
    reflective_audit_logprob_delta_threshold: float | None = None,
) -> SuiteResults:
    try:
        suite = get_suite(benchmark_version, suite_name)
        return benchmark_suite(
            suite,
            model,
            logdir,
            force_rerun,
            benchmark_version,
            user_tasks,
            injection_tasks,
            model_id,
            attack,
            defense,
            tool_delimiter,
            system_message_name,
            system_message,
            live,
            tool_output_format,
            reflective_audit_logprob_model,
            reflective_audit_verifier_model,
            reflective_audit_embedding_model,
            reflective_audit_top_k,
            reflective_audit_logprob_delta_threshold,
        )
    except Exception as exc:
        raise RuntimeError(f"Benchmark suite '{suite_name}' failed: {type(exc).__name__}: {exc}") from None


def show_results(suite_name: str, results: SuiteResults, show_security_results: bool):
    utility_results = results["utility_results"].values()
    avg_utility = sum(utility_results) / len(utility_results)

    print(f"Results for suite {suite_name}")
    print(f"Average utility: {avg_utility * 100:.2f}%")

    if show_security_results:
        passed_injection_tasks = sum(results["injection_tasks_utility_results"].values())
        total_injection_tasks = len(results["injection_tasks_utility_results"])
        print(f"\nPassed injection tasks as user tasks: {passed_injection_tasks}/{total_injection_tasks}")

        security_results = results["security_results"].values()
        avg_security = sum(security_results) / len(security_results)
        print(f"Average security: {avg_security * 100:.2f}%")


@click.command()
@click.option(
    "--model",
    default="openai/gpt-4o-mini-2024-07-18",
    type=click.Choice([value for value in ModelsEnum]),
    help="The language model to benchmark. Defaults to `openai/gpt-4o-mini-2024-07-18`.",
)
@click.option(
    "--model-id",
    type=str,
    default=None,
    help="The model id for local models.",
)
@click.option(
    "--benchmark-version",
    default="v1.2.2",
    type=str,
    help="The version of the benchmark to run. Defaults to `v1.2`.",
)
@click.option(
    "--tool-delimiter",
    type=str,
    default="tool",
    help="Which tool delimiter to use. Used for local models only.",
)
@click.option(
    "--logdir",
    default="./runs",
    type=Path,
    help="The directory to save logs. Defaults to `./runs`.",
)
@click.option(
    "--attack",
    type=str,
    default="important_instructions",
    help=f"The attack to use. `None` by default. It should be one of {ATTACKS}, or a custom attack defined and registered in of of the `--module-to-load`. If `None`, no attack is used.",
)
@click.option(
    "--defense",
    type=click.Choice([*DEFENSES, "none", "null", "no_defense", "no-defense"]),
    default="reflective_audit",
    help="The defense to use. Use `none` to disable defenses.",
)
@click.option(
    "--system-message-name",
    type=str,
    default=None,
    help="The name of the system message to use among the default ones in `data/system_messages.yaml`.",
)
@click.option(
    "--system-message",
    type=str,
    default=None,
    help="The system message to use (as a string). If provided, `--system-message-name` is ignored.",
)
@click.option(
    "--user-task",
    "-ut",
    "user_tasks",
    type=str,
    multiple=True,
    default=tuple(),
    help="The user tasks to benchmark. If not provided, all tasks in the suite are run.",
)
@click.option(
    "--injection-task",
    "-it",
    "injection_tasks",
    type=str,
    multiple=True,
    default=tuple(),
    help="The user tasks to benchmark. If not provided, all tasks in the suite are run.",
)
@click.option(
    "--suite",
    "-s",
    "suites",
    type=str,
    multiple=True,
    default=tuple(),
    help="The suite to benchmark. If not provided, dailylife is run.",
)
@click.option(
    "--max-workers",
    type=int,
    default=1,
    help="How many suites can be benchmarked in parallel through multithreading. Nothing is parallelized by default.",
)
@click.option(
    "--force-rerun",
    "-f",
    is_flag=True,
    help="Whether to re-run tasks that have already been run.",
)
@click.option(
    "--module-to-load",
    "-ml",
    "modules_to_load",
    type=str,
    multiple=True,
    default=tuple(),
    help="The modules to load before running the benchmark. Useful for registering custom attacks, defenses, or suites.",
)
@click.option(
    "--reflective-audit-logprob-model",
    type=str,
    default=None,
    help=(
        "Local causal LM path/name used for reflective_audit contrastive logprob scoring. "
        f"Defaults to REFLECTIVE_AUDIT_LOGPROB_MODEL or {DEFAULT_LOGPROB_MODEL}."
    ),
)
@click.option(
    "--reflective-audit-verifier-model",
    type=str,
    default=None,
    help=(
        "OpenAI-compatible model name used by reflective_audit to judge whether localized chunks contain IPI. "
        "Defaults to REFLECTIVE_AUDIT_VERIFIER_MODEL or the main benchmark model."
    ),
)
@click.option(
    "--reflective-audit-embedding-model",
    type=str,
    default=None,
    help=(
        "Embedding model path/name for --defense reflective_audit. "
        f"Defaults to REFLECTIVE_AUDIT_EMBEDDING_MODEL or {DEFAULT_EMBEDDING_MODEL}."
    ),
)
@click.option(
    "--reflective-audit-top-k",
    type=int,
    default=None,
    help=(
        "Number of chunks retrieved before reflective_audit attribution/verifier expansion. "
        "Defaults to REFLECTIVE_AUDIT_TOP_K or 3."
    ),
)
@click.option(
    "--reflective-audit-logprob-delta-threshold",
    type=float,
    default=DEFAULT_LOGPROB_DELTA_THRESHOLD,
    help=(
        "Minimum contrastive logprob delta required before reflective_audit sends localized chunks to the verifier. "
        f"Defaults to REFLECTIVE_AUDIT_LOGPROB_DELTA_THRESHOLD or {DEFAULT_LOGPROB_DELTA_THRESHOLD}."
    ),
)
def main(
    suites: tuple[str, ...],
    model: ModelsEnum,
    benchmark_version: str = "v1.2.2",
    logdir: Path = Path("./runs"),
    user_tasks: tuple[str, ...] = (),
    injection_tasks: tuple[str, ...] = (),
    model_id: str | None = None,
    attack: str | None = None,
    defense: str | None = None,
    tool_delimiter: str = "tool",
    system_message_name: str | None = None,
    system_message: str | None = None,
    max_workers: int = 1,
    force_rerun: bool = False,
    modules_to_load: tuple[str, ...] = (),
    reflective_audit_logprob_model: str | None = None,
    reflective_audit_verifier_model: str | None = None,
    reflective_audit_embedding_model: str | None = None,
    reflective_audit_top_k: int | None = None,
    reflective_audit_logprob_delta_threshold: float | None = None,
):
    for module in modules_to_load:
        importlib.import_module(module)

    if isinstance(attack, str) and attack.lower() in {"none", "null", "no_attack", "no-attack"}:
        attack = None
    if isinstance(defense, str) and defense.lower() in {"none", "null", "no_defense", "no-defense"}:
        defense = None

    if len(suites) == 0:
        suites = ("dailylife",)

    if len(suites) != 1:
        print(f"Benchmarking suites {', '.join(suites)}")

    if len(suites) != 1 and len(user_tasks) != 0:
        raise ValueError("A user task can be specified only when one suite is being executed")

    if max_workers == 1:
        results = {}
        for suite_name in suites:
            suite = get_suite(benchmark_version, suite_name)
            results[suite_name] = benchmark_suite(
                suite,
                model,
                logdir,
                benchmark_version=benchmark_version,
                user_tasks=user_tasks,
                injection_tasks=injection_tasks,
                model_id=model_id,
                attack=attack,
                defense=defense,
                tool_delimiter=tool_delimiter,
                system_message_name=system_message_name,
                system_message=system_message,
                force_rerun=force_rerun,
                reflective_audit_logprob_model=reflective_audit_logprob_model,
                reflective_audit_verifier_model=reflective_audit_verifier_model,
                reflective_audit_embedding_model=reflective_audit_embedding_model,
                reflective_audit_top_k=reflective_audit_top_k,
                reflective_audit_logprob_delta_threshold=reflective_audit_logprob_delta_threshold,
            )
        combined_results = SuiteResults(utility_results={}, security_results={}, injection_tasks_utility_results={})
        for suite_name, result in results.items():
            show_results(suite_name, result, attack is not None)
            for user_task, injection_task in result["utility_results"]:
                new_user_task = suite_name + "_" + user_task
                combined_results["utility_results"][(new_user_task, injection_task)] = result["utility_results"][
                    (user_task, injection_task)
                ]
            for user_task, injection_task in result["security_results"]:
                new_user_task = suite_name + "_" + user_task
                combined_results["security_results"][(new_user_task, injection_task)] = result["security_results"][
                    (user_task, injection_task)
                ]
            for injection_task in result["injection_tasks_utility_results"]:
                new_injection_task = suite_name + "_" + injection_task
                combined_results["injection_tasks_utility_results"][new_injection_task] = result[
                    "injection_tasks_utility_results"
                ][injection_task]
        show_results("combined", combined_results, attack is not None)
        exit(0)

    with Pool(max_workers) as p:
        arguments = zip(
            suites,
            repeat(model),
            repeat(logdir),
            repeat(force_rerun),
            repeat(benchmark_version),
            repeat(user_tasks),
            repeat(injection_tasks),
            repeat(model_id),
            repeat(attack),
            repeat(defense),
            repeat(tool_delimiter),
            repeat(system_message_name),
            repeat(system_message),
            repeat(None),
            repeat(None),
            repeat(reflective_audit_logprob_model),
            repeat(reflective_audit_verifier_model),
            repeat(reflective_audit_embedding_model),
            repeat(reflective_audit_top_k),
            repeat(reflective_audit_logprob_delta_threshold),
        )
        results = p.starmap(benchmark_suite_from_name, arguments)

    for suite_name, result in zip(suites, results):
        show_results(suite_name, result, attack is not None)


if __name__ == "__main__":
    format = "%(message)s"
    logging.basicConfig(
        format=format,
        level=logging.INFO,
        datefmt="%H:%M:%S",
        handlers=[RichHandler(show_path=False, markup=True)],
    )
    main()
