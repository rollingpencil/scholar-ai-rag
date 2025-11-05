from models.models import AblationConfig, QAResultModel, QueryAnswerPair
from service.judge_svc import evaluate_response
from service.query_svc import query
from utils.logger import log


async def evaluate_qns_ans_with_judge(
    query_text: str,
    expected_ans: str,
    ablation_config: AblationConfig = None
) -> QAResultModel:
    graph_response = await query(query_text, ablation_config=ablation_config)

    qapair = QueryAnswerPair(
        query=query_text,
        expected_answer=expected_ans,
        actual_answer=graph_response.answer,
        actual_reasoning=graph_response.reasoning,
    )
    log.debug(f"QA Pair: {qapair}")

    evaluation = await evaluate_response(qapair)
    return QAResultModel(qapair=qapair, evaluation=evaluation)
