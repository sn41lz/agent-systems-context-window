"""A caller owns task-state updates; the compiler owns bounded context assembly."""

from context_window.engine import ContextEngine
from context_window.models import Budget, FactUpdate, Turn

engine = ContextEngine(budget=Budget(context_tokens=8192, output_reserve=2048))
engine.ingest(
    Turn(
        "incident-open", "Investigate checkout latency; EU data residency applies.", "Understood."
    ),
    (FactUpdate("residency", "Customer records must remain in the EU", expected_revision=0),),
)
request = engine.compile("What evidence is still needed before a safe rollout?")
print(request.metadata())
# Optional, billable integration with an explicitly chosen supported model:
#
# from context_window.live import OpenAIStream
# from context_window.modern import RequestOptions
# provider = OpenAIStream(model, options=RequestOptions(preflight=True, reasoning_effort="low"))
# try:
#     sample = await provider.generate(request, max_output_tokens=engine.budget.output_reserve)
# finally:
#     await provider.aclose()
#
# A Turn is the manual text/tool replay format. Preserve all response output items
# separately when continuing real reasoning/tool conversations; see docs/current-models.md.
