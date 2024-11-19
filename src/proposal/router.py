from fastapi import APIRouter

from schemas import ProposalEvaluationRequestDto, SummaryGenerationRequestDto
from service import proposal_evaluation, summary_generation

router = APIRouter()


@router.post("/ai/proposal/evaluation")
def proposal_evaluation_router(request: ProposalEvaluationRequestDto):
    return proposal_evaluation(request)


@router.post("/ai/proposal/summary")
def summary_generation_router(request: SummaryGenerationRequestDto):
    return summary_generation(request)
