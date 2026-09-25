"""The tools the demo deployment offers its agent.

The platform itself ships no capabilities — a deployment declares what its
agents may do — so these are what a deployment's tool module looks like, and
they are the only tools in this repository with a business meaning. They span
the risk range on purpose: reading an invoice is ``READ``, annotating one is
``WRITE``, and paying one is ``CRITICAL``, which is what makes the default
``EAP_AGENT_AUTO_APPROVE_UP_TO=read`` produce a visible approval pause instead
of a paragraph claiming there would be one.

Handlers return compact JSON rather than prose. Two reasons: a model parses a
result far more reliably than it parses a sentence, and a deterministic string
keeps the demo's output stable enough to put in a README.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from enterprise_agent_platform.demo.data import AS_OF, Invoice, Ledger, LedgerError
from enterprise_agent_platform.tools.models import RiskLevel, Tool, ToolExecutionError

INVOICE_ID_PATTERN = r"^INV-\d{4}$"
AMOUNT_PATTERN = r"^\d{1,9}\.\d{2}$"
"""Money crosses the model boundary as an exact decimal string, never a JSON
number: JSON numbers are IEEE doubles, and a payment amount that has been
through a float is no longer the amount on the invoice. The pattern also keeps
the tool's schema a plain constrained string, which is what a model reproduces
most reliably."""


class LookupInvoiceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invoice_id: str = Field(
        pattern=INVOICE_ID_PATTERN, description="Invoice identifier, e.g. INV-1043."
    )


class ListOverdueInvoicesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supplier: str | None = Field(
        default=None, max_length=120, description="Optional case-insensitive supplier filter."
    )
    limit: int = Field(default=10, ge=1, le=50, description="Maximum invoices to return.")


class AddInvoiceNoteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invoice_id: str = Field(pattern=INVOICE_ID_PATTERN, description="Invoice to annotate.")
    note: str = Field(
        min_length=1, max_length=500, description="Short note to append to the invoice."
    )


class PayInvoiceArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invoice_id: str = Field(pattern=INVOICE_ID_PATTERN, description="Invoice to settle.")
    amount_eur: str = Field(
        pattern=AMOUNT_PATTERN,
        description=(
            "Amount to pay in EUR, as an exact decimal string with two places, e.g. "
            '"1284.50". It must equal the invoice total exactly: look the invoice up '
            "first rather than estimating it."
        ),
    )

    @property
    def amount(self) -> Decimal:
        """The amount as exact arithmetic. The pattern guarantees this parses."""
        return Decimal(self.amount_eur)


def build_demo_tools(ledger: Ledger | None = None) -> tuple[Tool[Any], ...]:
    """Build the demo tool set over a ledger, defaulting to a fresh one.

    The ledger is a parameter so a caller can inspect what the agent actually
    did — the demo walkthrough and the tests both assert against it, rather than
    against the model's own account of itself.
    """
    ledger = ledger if ledger is not None else Ledger()

    async def lookup_invoice(args: LookupInvoiceArgs) -> str:
        with _reporting_to_the_model():
            invoice = ledger.get(args.invoice_id)
        return _dump(
            {
                **_invoice_fields(invoice),
                "paid": args.invoice_id in ledger.payments,
                "notes": ledger.notes.get(args.invoice_id, ()),
            }
        )

    async def list_overdue_invoices(args: ListOverdueInvoicesArgs) -> str:
        invoices = ledger.overdue(supplier=args.supplier)[: args.limit]
        return _dump(
            {
                "as_of": AS_OF.isoformat(),
                "count": len(invoices),
                "invoices": [_invoice_fields(invoice) for invoice in invoices],
            }
        )

    async def add_invoice_note(args: AddInvoiceNoteArgs) -> str:
        with _reporting_to_the_model():
            notes = ledger.add_note(args.invoice_id, args.note)
        return _dump({"invoice_id": args.invoice_id, "notes": notes})

    async def pay_invoice(args: PayInvoiceArgs) -> str:
        with _reporting_to_the_model():
            payment = ledger.pay(args.invoice_id, args.amount)
        return _dump(
            {
                "invoice_id": payment.invoice_id,
                "amount_eur": str(payment.amount_eur),
                "reference": payment.reference,
                "status": "paid",
            }
        )

    return (
        Tool(
            name="lookup_invoice",
            description=(
                "Look up one supplier invoice by id and return its supplier, amount, due "
                "date, payment status and notes."
            ),
            arguments=LookupInvoiceArgs,
            risk=RiskLevel.READ,
            handler=lookup_invoice,
        ),
        Tool(
            name="list_overdue_invoices",
            description=(
                "List unpaid invoices whose due date has passed, most overdue first, "
                "optionally filtered to one supplier."
            ),
            arguments=ListOverdueInvoicesArgs,
            risk=RiskLevel.READ,
            handler=list_overdue_invoices,
        ),
        Tool(
            name="add_invoice_note",
            description=(
                "Append a short note to an invoice. Use it to record a finding; it does not "
                "change the amount, the due date or the payment status."
            ),
            arguments=AddInvoiceNoteArgs,
            risk=RiskLevel.WRITE,
            handler=add_invoice_note,
        ),
        Tool(
            name="pay_invoice",
            description=(
                "Pay a supplier invoice in full. This moves money and cannot be undone: the "
                "amount must match the invoice exactly, and an invoice can only be paid once."
            ),
            arguments=PayInvoiceArgs,
            risk=RiskLevel.CRITICAL,
            handler=pay_invoice,
        ),
    )


@contextmanager
def _reporting_to_the_model() -> Iterator[None]:
    """Translate a ledger failure into the one exception type a model may read.

    ``ToolExecutionError`` is the registry's contract for "this is an outcome in
    the tool's own domain, hand the text back so the run can recover". Anything
    else escaping a handler is treated as a defect and replaced with a generic
    message — which is the right thing to do to an unexpected error, and the
    wrong thing to do to "no invoice with that id".
    """
    try:
        yield
    except LedgerError as exc:
        raise ToolExecutionError(str(exc)) from exc


def _invoice_fields(invoice: Invoice) -> dict[str, Any]:
    return {
        "invoice_id": invoice.id,
        "supplier": invoice.supplier,
        "amount_eur": str(invoice.amount_eur),
        "due_on": invoice.due_on.isoformat(),
        "days_overdue": invoice.days_overdue(),
        "description": invoice.description,
    }


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)
