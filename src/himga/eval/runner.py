"""Evaluation runner: drives the predict loop and collects EvalResult objects."""

from __future__ import annotations

from dataclasses import dataclass

from tqdm import tqdm

from himga.agent.base import BaseAgent
from himga.data.schema import QuestionType, Sample


@dataclass
class EvalResult:
    """Prediction record for one QA pair.

    Parameters
    ----------
    sample_id : str
        Identifier of the parent :class:`~himga.data.schema.Sample`.
    question_id : str
        Identifier of the :class:`~himga.data.schema.QAPair`.
    question_type : QuestionType
        Category used for per-type metric aggregation.
    question : str
        The question text.
    ground_truth : str
        Gold-standard answer.
    prediction : str
        Agent-generated answer.
    """

    sample_id: str
    question_id: str
    question_type: QuestionType
    question: str
    ground_truth: str
    prediction: str


def run_eval(
    dataset: list[Sample],
    agent: BaseAgent,
    *,
    show_progress: bool = True,
) -> list[EvalResult]:
    """Run the full evaluation loop over *dataset*.

    For each sample the agent's memory is reset, the sample history is ingested,
    and every QA pair is answered.  Results are collected without calling judge or
    metrics — those are intentionally deferred so that expensive judge calls can be
    batched or cached after the fact.

    Parameters
    ----------
    dataset : list[Sample]
        Samples to evaluate.
    agent : BaseAgent
        Agent whose ``memory`` is reset per sample and whose ``answer`` method
        is called for each question.
    show_progress : bool
        Whether to display a tqdm progress bar.

    Returns
    -------
    list[EvalResult]
        One :class:`EvalResult` per QA pair, in dataset order.
    """
    results: list[EvalResult] = []
    for sample in tqdm(dataset, disable=not show_progress):
        agent.memory.reset()
        agent.ingest_sample(sample)
        for qa in sample.qa_pairs:
            prediction = agent.answer(qa.question)
            results.append(
                EvalResult(
                    sample_id=sample.sample_id,
                    question_id=qa.question_id,
                    question_type=qa.question_type,
                    question=qa.question,
                    ground_truth=qa.answer,
                    prediction=prediction,
                )
            )
    return results
