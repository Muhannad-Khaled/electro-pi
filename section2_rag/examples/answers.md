# Example Q&A — real pipeline output

All four transcripts below are **verbatim output** from
`python -m src.cli --verbose "<question>"` with the final pipeline
configuration (LLM: `gemini-3.1-flash-lite`, embeddings:
`gemini-embedding-001`, TOP_K=8). Nothing is hand-written or edited.

---

## 1. Conditional refund logic

**Question:** *I cancelled my order 8 minutes after placing it. Do I get a full refund?*

**Retrieval & grader decision:**

```
--- Retrieval ---
  [1] refund_policy.md#1  (L2 distance: 0.5487)
  [2] refund_policy.md#3  (L2 distance: 0.6515)
  [3] refund_policy.md#2  (L2 distance: 0.6665)
  [4] refund_policy.md#4  (L2 distance: 0.6950)
  [5] account_and_orders.md#1  (L2 distance: 0.6977)
  [6] delivery_and_tracking.md#4  (L2 distance: 0.7017)
  [7] delivery_and_tracking.md#1  (L2 distance: 0.7057)
  [8] refund_policy.md#0  (L2 distance: 0.7255)
--- Grader: relevant chunks: [1] ---
--- Route: answer ---
```

**Answer:**

> No, you do not get a full refund.
>
> Whether you receive a partial refund depends on the status of your order:
>
> *   **If the restaurant has not yet accepted the order:** The context does not explicitly state the refund policy for cancellations made after 5 minutes if the restaurant has not yet accepted the order; it only specifies that the "5-minute window no longer applies" once the restaurant marks the order as "Accepted" [1].
> *   **If the restaurant has accepted the order (but the courier has not picked it up):** You are eligible for a refund of "**50%** of the order subtotal" [1]. In this scenario, the delivery fee and service fee are "non-refundable" [1].
> *   **If the courier has already picked up the food:** You are "not eligible for a refund" [1]. In this case, you should "contact support to report the issue rather than cancelling" [1].
>
> Sources:
>   [1] refund_policy.md#1

**Citations:** `refund_policy.md#1`

The model correctly applies the conditional logic (5-minute window, restaurant
acceptance, courier pickup) and — notably — flags honestly that the policy text
does not explicitly cover the "past 5 minutes but not yet accepted" edge case,
rather than inventing a rule for it.

---

## 2. Cross-document: cash payment + missing items

**Question:** *I paid cash for an order and half the items were missing. How do I get my money back?*

**Retrieval & grader decision:**

```
--- Retrieval ---
  [1] refund_policy.md#2  (L2 distance: 0.6003)
  [2] refund_policy.md#1  (L2 distance: 0.6371)
  [3] refund_policy.md#4  (L2 distance: 0.6494)
  [4] refund_policy.md#0  (L2 distance: 0.6863)
  [5] account_and_orders.md#1  (L2 distance: 0.6962)
  [6] delivery_and_tracking.md#7  (L2 distance: 0.7007)
  [7] refund_policy.md#3  (L2 distance: 0.7233)
  [8] payments_and_billing.md#1  (L2 distance: 0.7510)
--- Grader: relevant chunks: [1, 7] ---
--- Route: answer ---
```

**Answer:**

> To determine how you get your money back, we must first clarify the value of the missing items relative to your total order:
>
> *   **If 50% or less of the order value is missing or incorrect:** You will be refunded only for the affected items, which includes the item's price plus its "proportional share of the service fee" [1]. The delivery fee will not be refunded [1].
> *   **If more than 50% of the order value is missing or incorrect:** You will be refunded the full order amount, including the delivery fee [1]. Additionally, you will be offered a "15% credit on their next order as a goodwill gesture" [1].
>
> **How you will receive the refund:**
> Because you paid via cash on delivery, your refund will be issued as "Wasl Wallet credit only" [2]. Wasl Eats does not issue cash refunds directly to customers because there is no card transaction to reverse [2]. Once the refund is approved, the credit will appear in your Wasl Wallet instantly, "within 60 seconds of approval" [2].
>
> **Note:** The provided context does not specify the exact process for initiating the refund request (e.g., where to click in the app or how to contact support to report the missing items).
>
> Sources:
>   [1] refund_policy.md#2
>   [2] refund_policy.md#3

**Citations:** `refund_policy.md#2`, `refund_policy.md#3`

**Honest note on citation span:** this question was designed to force
cross-document coupling, and retrieval does span both files — the cash
payment-method chunk `payments_and_billing.md#1` is in the retrieved
candidate set (rank 8). The grader, however, excludes it, because both facts
that decide the answer (the >50%-missing rule and the cash→wallet-credit
rule) live in `refund_policy.md`; the payments chunk only describes cash
*eligibility at checkout* (EGP 800 limit, area support), which is not needed
to answer "how do I get my money back". I verified this exclusion with a
stronger grader model (`gemini-3-flash-preview`), which reached the same
verdict. I chose to keep the honest grader rather than prompt-engineer it
into citing a chunk the answer does not use — the guardrail matters more
than the citation count.

---

## 3. Body-level detail: address change after pickup

**Question:** *Can I change my delivery address after the courier picks up the order?*

**Retrieval & grader decision:**

```
--- Retrieval ---
  [1] account_and_orders.md#2  (L2 distance: 0.4672)
  [2] account_and_orders.md#1  (L2 distance: 0.5791)
  [3] delivery_and_tracking.md#6  (L2 distance: 0.6034)
  [4] delivery_and_tracking.md#5  (L2 distance: 0.6476)
  [5] delivery_and_tracking.md#2  (L2 distance: 0.6696)
  [6] refund_policy.md#1  (L2 distance: 0.6852)
  [7] delivery_and_tracking.md#4  (L2 distance: 0.6919)
  [8] refund_policy.md#4  (L2 distance: 0.7151)
--- Grader: relevant chunks: [1] ---
--- Route: answer ---
```

**Answer:**

> Once the courier is en route (the "On the way" stage), you can no longer change the address through the app settings [1]. Instead, you must call the courier directly using the in-app call button [1]. Please note that any change at this stage is at the courier's discretion [1].
>
> If you need to change to a fully different address, the correct action is to cancel your current order and place a new one [1]. Be aware that canceling is subject to the refund policy's cancellation rules [1].
>
> Sources:
>   [1] account_and_orders.md#2

**Citations:** `account_and_orders.md#2`

The decisive facts (in-app call, courier's discretion, cancel-and-reorder for
a fully different address) are body-level details inside the "Changing the
Delivery Address" section — not derivable from any heading.

---

## 4. Out of scope — abstain guardrail

**Question:** *What's the best pizza place in Cairo?*

**Retrieval & grader decision:**

```
--- Retrieval ---
  [1] payments_and_billing.md#4  (L2 distance: 0.7521)
  [2] payments_and_billing.md#1  (L2 distance: 0.7777)
  [3] delivery_and_tracking.md#0  (L2 distance: 0.8320)
  [4] payments_and_billing.md#0  (L2 distance: 0.8353)
  [5] refund_policy.md#0  (L2 distance: 0.8391)
  [6] delivery_and_tracking.md#7  (L2 distance: 0.8405)
  [7] delivery_and_tracking.md#1  (L2 distance: 0.8497)
  [8] payments_and_billing.md#3  (L2 distance: 0.8660)
--- Grader: NO relevant chunks ---
--- Route: abstain (no generation LLM call) ---
```

**Answer:**

> I couldn't find information about this in the support documents. This question may be outside the scope of what I can answer reliably.

**Citations:** none

FAISS still returns the 8 "closest" chunks — nearest-neighbor search always
returns *something* — but the grader marks zero of them relevant, the graph
routes to the abstain node, and the fixed message is returned **without any
generation LLM call**. This is exactly the failure mode a raw
similarity-score threshold cannot handle reliably.
