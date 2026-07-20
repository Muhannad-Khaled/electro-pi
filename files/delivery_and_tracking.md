# Wasl Eats — Delivery and Order Tracking

This document covers how deliveries are timed, how the tracking screen works,
and what to do when an order appears stuck.

## Delivery Time Estimates

Every order shows an estimated delivery time (ETA) at checkout. The ETA is
calculated from three factors: the restaurant's current preparation time, the
distance between the restaurant and the delivery address, and current traffic
conditions on the courier's likely route.

The ETA is a range, not a promise. Actual delivery may fall outside the range
during peak hours (typically 12:00–14:00 and 19:00–22:00 local time) or during
weather disruptions. When the ETA is exceeded by more than **20 minutes**, the
customer is automatically eligible for a delivery-fee refund, applied as Wasl
Wallet credit without needing to contact support.

## Tracking Stages

The tracking screen shows the order moving through five stages:

1. **Placed** — the order has been sent to the restaurant but not yet accepted.
2. **Accepted** — the restaurant has confirmed and started preparing the food.
3. **Preparing** — food is being cooked or assembled.
4. **Ready for pickup** — food is packaged and waiting for a courier.
5. **On the way** — a courier has collected the order and is en route.

The map view only becomes live during the "On the way" stage. Before that, the
courier location is not shown because no specific courier has been assigned.

## Stuck Orders

An order is considered stuck when it stays in the same stage longer than
expected. Common causes and expected wait times:

- **Stuck at "Placed" for more than 3 minutes:** the restaurant has not yet
  seen the order. The system will auto-cancel and fully refund the order if
  the restaurant does not accept within **8 minutes** of placement.
- **Stuck at "Preparing" for more than 25 minutes:** usually a busy kitchen.
  Support can contact the restaurant on the customer's behalf, but cannot
  cancel the order at this stage without the restaurant's agreement, since the
  food may already be prepared.
- **Stuck at "Ready for pickup" for more than 15 minutes:** courier assignment
  delay. The system automatically increases courier incentives after 10
  minutes to attract a nearby courier.

If an order remains stuck at "On the way" for more than 45 minutes past the
original ETA, customers should call the courier directly using the in-app call
button. Courier phone numbers are masked for privacy — direct dialling of
personal numbers is not possible.

## Courier Contact Rules

The courier can be contacted only during the "On the way" stage. Before pickup,
no courier has been assigned yet. The in-app chat with the courier is available
for the entire delivery, but voice calls are limited to a maximum of three per
order to prevent misuse.

Couriers are instructed to wait a maximum of **10 minutes** at the delivery
address if the customer does not respond. After 10 minutes, the courier may
mark the order as undeliverable and leave. In this case the customer is
charged the full order amount and no refund is issued (see refund policy for
address-error rules).

## Contactless and Special Delivery

Customers can request contactless delivery in the delivery notes. When enabled,
the courier will leave the order at the door and confirm delivery by photo.
Contactless delivery is not available for orders paid by cash, because payment
must be collected in person.
