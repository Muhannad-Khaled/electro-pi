# Wasl Eats — Payments and Billing

This document covers accepted payment methods, how failed payments are handled,
and the rules for promo codes and invoicing.

## Accepted Payment Methods

Wasl Eats accepts the following payment methods:

- Credit and debit cards (Visa, Mastercard, Meeza).
- Apple Pay and Google Pay.
- Wasl Wallet balance (funded by card top-up or promotional credits).
- Cash on delivery, available only for orders under **EGP 800** and only in
  areas where the courier network supports cash handling. Cash is not
  available for orders scheduled more than 4 hours in advance.

American Express is not currently supported. Bank transfers and cryptocurrency
are not accepted for individual orders. Corporate customers on an invoicing
plan are handled separately (see the Corporate Billing section below).

## Failed Payments

If a card payment fails at checkout, the app automatically retries the
transaction **once** after 15 seconds. If the retry also fails, the customer
is prompted to select a different payment method or update the card details.
The order is not placed until a payment succeeds.

For scheduled orders, payment is authorized at the time of scheduling and
captured 30 minutes before the scheduled delivery time. If capture fails
(for example, because the card has expired between scheduling and delivery),
the customer receives a push notification and has 15 minutes to update the
payment method before the order is cancelled.

## Promo Codes

Only **one** promo code can be applied per order. Promo codes cannot be
stacked with each other, but they **can** be combined with Wasl Wallet credit —
the promo discount is applied first, then wallet credit covers the remainder.

Promo codes are tied to the account, not the device. Using multiple accounts
to redeem a single-use promo code is against the terms of service and may
result in account suspension.

Promo codes expire at midnight local time on the stated expiry date. Codes
have a minimum order value which is shown when the code is applied — if the
order value drops below that minimum during editing (for example, by removing
an item), the code is automatically removed and must be re-applied if the
order value increases again.

## Service Fee and VAT

Every order includes a service fee of **7%** of the food subtotal, capped at
EGP 25. VAT is applied to the food subtotal, the delivery fee, and the service
fee at the standard rate. Both the service fee and VAT are shown as separate
line items on the receipt.

Tips to couriers, when added in the app, are passed to the courier in full
and are not subject to service fee or VAT.

## Invoicing and Receipts

A receipt is automatically emailed to the account email after each successful
order. Receipts can also be downloaded from the order history screen for up to
**12 months** after the order date. Older receipts can be requested through
support, but may take up to 5 business days to retrieve.

Corporate customers on the Wasl for Business plan receive a monthly
consolidated invoice instead of per-order receipts. The invoice is issued on
the 1st of each month and payment is due within 15 days. Late payment may
result in the account being paused until the invoice is settled.
