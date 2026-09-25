"""Synthetic accounts-payable data the demo tools operate on.

Everything here is invented: the suppliers, the invoice numbers, the amounts.
Nothing reads a file, opens a socket, or touches an employer's data. It exists
so the platform can be *run* rather than described — a risk gate, an approval
pause and a resumed run are only convincing when a reader can watch them
happen against something.

Two decisions keep the demo reproducible. The dataset is frozen at a fixed
``AS_OF`` date, so "overdue" means the same thing today as it will next year;
and the mutable half lives in a ``Ledger`` instance rather than in module
globals, so a payment made by one test cannot be observed by the next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

AS_OF = date(2026, 3, 31)
"""The date the demo dataset is frozen at; what the demo tools call 'today'.

Real wall-clock time would make the overdue list drift and eventually empty
out, so the fixture carries its own present.
"""


class LedgerError(Exception):
    """A demo-domain failure: unknown invoice, double payment, wrong amount.

    Separate from ``ToolExecutionError`` because this module knows nothing about
    tools. ``demo.tools`` translates it, which is also the boundary at which the
    text becomes something a model is allowed to read.
    """


@dataclass(frozen=True, slots=True)
class Invoice:
    """One supplier invoice. Amounts are ``Decimal``: money is not a float."""

    id: str
    supplier: str
    amount_eur: Decimal
    due_on: date
    description: str

    def is_overdue(self, as_of: date = AS_OF) -> bool:
        return self.due_on < as_of

    def days_overdue(self, as_of: date = AS_OF) -> int:
        return max((as_of - self.due_on).days, 0)


@dataclass(frozen=True, slots=True)
class Payment:
    """The record a critical tool leaves behind when it moves money."""

    invoice_id: str
    amount_eur: Decimal
    reference: str


DEMO_INVOICES: tuple[Invoice, ...] = (
    Invoice(
        id="INV-1043",
        supplier="Northwind Logistics",
        amount_eur=Decimal("1284.50"),
        due_on=date(2026, 3, 10),
        description="Freight consolidation, February",
    ),
    Invoice(
        id="INV-1044",
        supplier="Northwind Logistics",
        amount_eur=Decimal("318.00"),
        due_on=date(2026, 4, 15),
        description="Palletisation surcharge, March",
    ),
    Invoice(
        id="INV-1051",
        supplier="Halden Facilities",
        amount_eur=Decimal("7420.00"),
        due_on=date(2026, 2, 28),
        description="Quarterly facilities management",
    ),
    Invoice(
        id="INV-1052",
        supplier="Cartwright Print",
        amount_eur=Decimal("96.75"),
        due_on=date(2026, 3, 29),
        description="Compliance leaflets, 2,000 units",
    ),
)


@dataclass(slots=True)
class Ledger:
    """The demo's mutable state: which invoices are paid, and their notes.

    Scoped to one instance so the demo has no process-wide memory. Payments are
    keyed by invoice id, which is what makes paying twice detectable — the guard
    the approval demo depends on being real rather than narrated.
    """

    invoices: tuple[Invoice, ...] = DEMO_INVOICES
    payments: dict[str, Payment] = field(default_factory=dict)
    notes: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def get(self, invoice_id: str) -> Invoice:
        for invoice in self.invoices:
            if invoice.id == invoice_id:
                return invoice
        raise LedgerError(f"no invoice with id '{invoice_id}'")

    def overdue(self, *, supplier: str | None = None, as_of: date = AS_OF) -> tuple[Invoice, ...]:
        overdue = [
            invoice
            for invoice in self.invoices
            if invoice.is_overdue(as_of)
            and invoice.id not in self.payments
            and (supplier is None or supplier.lower() in invoice.supplier.lower())
        ]
        # Most overdue first: the order an approver would work the list in.
        return tuple(sorted(overdue, key=lambda invoice: invoice.due_on))

    def pay(self, invoice_id: str, amount_eur: Decimal) -> Payment:
        """Settle an invoice, refusing anything that does not match the record.

        The amount is checked against the ledger rather than trusted: it reaches
        this call as an argument a language model wrote, and a critical tool
        that accepts whatever it is handed is a gate with no lock in it.
        """
        invoice = self.get(invoice_id)
        if invoice_id in self.payments:
            existing = self.payments[invoice_id]
            raise LedgerError(
                f"invoice '{invoice_id}' was already paid under reference {existing.reference}"
            )
        if amount_eur != invoice.amount_eur:
            raise LedgerError(
                f"amount {amount_eur} does not match invoice '{invoice_id}' "
                f"({invoice.amount_eur} EUR)"
            )
        payment = Payment(
            invoice_id=invoice_id,
            amount_eur=amount_eur,
            reference=f"PAY-{invoice_id.removeprefix('INV-')}",
        )
        self.payments[invoice_id] = payment
        return payment

    def add_note(self, invoice_id: str, note: str) -> tuple[str, ...]:
        self.get(invoice_id)
        notes = (*self.notes.get(invoice_id, ()), note)
        self.notes[invoice_id] = notes
        return notes
