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

    Uses a two-phase approach: first ingest all samples and collect LLM
    requests, then fire all requests as a single :meth:`~himga.llm.BaseLLMClient.batch_chat`
    call.  This allows concurrent async execution in the underlying client.

    Parameters
    ----------
    dataset : list[Sample]
        Samples to evaluate.
    agent : BaseAgent
        Agent whose ``memory`` is reset per sample and whose messages are built
        for each question.
    show_progress : bool
        Whether to display a tqdm progress bar.

    Returns
    -------
    list[EvalResult]
        One :class:`EvalResult` per QA pair, in dataset order.
    """
    # Phase 1: ingest each sample and collect the LLM request for every QA pair.
    pending: list[tuple[str, str, QuestionType, str, str]] = []
    requests: list[dict] = []

    for sample in tqdm(dataset, disable=not show_progress):
        agent.memory.reset()
        agent.ingest_sample(sample)
        for qa in sample.qa_pairs:
            context = agent.memory.retrieve(qa.question)
            messages = agent._build_messages(qa.question, context)
            pending.append(
                (sample.sample_id, qa.question_id, qa.question_type, qa.question, qa.answer)
            )
            requests.append({"messages": messages})

    if not pending:
        return []

    # Phase 2: single batched async call to the LLM.
    predictions = agent.llm.batch_chat(requests)

    # Phase 3: assemble EvalResult objects.
    return [
        EvalResult(
            sample_id=sample_id,
            question_id=question_id,
            question_type=question_type,
            question=question,
            ground_truth=ground_truth,
            prediction=pred,
        )
        for (sample_id, question_id, question_type, question, ground_truth), pred in zip(
            pending, predictions
        )
    ]
