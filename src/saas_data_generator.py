"""
Synthetic B2B SaaS + finance data generator for a Microsoft Fabric portfolio project.

Run it in a Fabric notebook with your Lakehouse attached as the default lakehouse.
It writes raw CSVs to Files/bronze/saas/ (falls back to ./output/bronze/saas locally).

Tables produced (all "bronze": raw, deliberately not analysis-ready):
  customers.csv           one row per customer
  subscriptions_monthly   one row per customer per ACTIVE month (a churned customer simply stops appearing)
  invoices.csv            billed and collected amounts, with payment dates
  marketing_spend.csv     spend by month and channel (for CAC)
  chart_of_accounts.csv   account master for the P&L
  financials.csv          Actual and Budget amounts by month and account

Design notes
  * Deterministic: every customer and month has its own seeded RNG, so re-running with a
    later AS_OF reproduces history exactly and only ADDS new months. Use this in Phase 2 to
    simulate new batches arriving and to test watermark-based incremental loads.
  * Churn, new, expansion and contraction MRR are NOT labelled in bronze. You derive them in
    silver/gold with window functions (LAG / LEAD), which is the SQL showpiece of the project.
  * Budget is derived from Actual with a yearly bias, so the demo has a believable story:
    revenue beats budget in 2024, misses in 2025 and 2026.
"""
import os
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ config
SEED = 42
N_CUSTOMERS = 2500                  # customers that could ever sign up
START_MONTH = "2023-01-01"
LAST_SIGNUP_MONTH = "2027-06-01"    # fixed ceiling so reruns stay deterministic
AS_OF = "2026-10-01"                # generate data up to this month; move forward to simulate a new batch
OUT_DIR = ("/lakehouse/default/Files/bronze/saas"
           if os.path.exists("/lakehouse/default") else "./output/bronze/saas")

MONTHS = pd.date_range(START_MONTH, LAST_SIGNUP_MONTH, freq="MS")
AS_OF_IDX = MONTHS.get_loc(pd.Timestamp(AS_OF))
AS_OF_END = pd.Timestamp(AS_OF) + pd.offsets.MonthEnd(0)

PLANS = ["Starter", "Pro", "Enterprise"]
PRICE_PER_SEAT = {"Starter": 15, "Pro": 35, "Enterprise": 60}
SEAT_RANGE = {"Starter": (3, 12), "Pro": (10, 60), "Enterprise": (50, 300)}
CHURN = {"Starter": 0.035, "Pro": 0.018, "Enterprise": 0.008}       # base monthly churn
UPGRADE = {"Starter": 0.020, "Pro": 0.012, "Enterprise": 0.0}
DOWNGRADE = {"Starter": 0.0, "Pro": 0.008, "Enterprise": 0.006}
NEXT_PLAN = {"Starter": "Pro", "Pro": "Enterprise"}
PREV_PLAN = {"Pro": "Starter", "Enterprise": "Pro"}

CHANNELS = ["Paid Search", "Outbound", "Partner", "Organic", "Events"]
CAC = {"Paid Search": 900, "Outbound": 1800, "Partner": 600, "Organic": 150, "Events": 1400}
REV_BUDGET_BIAS = {2023: 1.03, 2024: 0.98, 2025: 1.05, 2026: 1.08, 2027: 1.08}

ACCOUNTS = [
    (4000, "Subscription Revenue", "Revenue"),
    (4100, "Services Revenue", "Revenue"),
    (5000, "Hosting", "COGS"),
    (5100, "Customer Support", "COGS"),
    (5200, "Payment Processing", "COGS"),
    (6000, "Sales & Marketing", "Opex"),
    (6100, "Research & Development", "Opex"),
    (6200, "General & Admin", "Opex"),
]


# ------------------------------------------------------------------ customers
def build_customers():
    rng = np.random.default_rng(SEED)
    n = N_CUSTOMERS
    weights = np.linspace(1.0, 2.5, len(MONTHS))       # signups accelerate over time
    weights = weights / weights.sum()
    signup_idx = rng.choice(len(MONTHS), size=n, p=weights)

    segment = rng.choice(["SMB", "Mid-Market", "Enterprise"], size=n, p=[0.65, 0.28, 0.07])
    annual_p = pd.Series(segment).map({"SMB": 0.15, "Mid-Market": 0.40, "Enterprise": 0.70}).to_numpy()
    plan_mix = {"SMB": [0.70, 0.30, 0.00],
                "Mid-Market": [0.15, 0.70, 0.15],
                "Enterprise": [0.00, 0.20, 0.80]}
    initial_plan = [str(rng.choice(PLANS, p=plan_mix[s])) for s in segment]
    initial_seats = [int(rng.integers(*SEAT_RANGE[p])) for p in initial_plan]

    df = pd.DataFrame({
        "customer_id": [f"C{i + 1:05d}" for i in range(n)],
        "company_name": [f"Company {i + 1:05d}" for i in range(n)],
        "segment": segment,
        "region": rng.choice(["North America", "EMEA", "APAC", "LATAM"], size=n, p=[0.45, 0.30, 0.18, 0.07]),
        "industry": rng.choice(["Software", "Financial Services", "Healthcare",
                                "Retail", "Manufacturing", "Education"], size=n),
        "acquisition_channel": rng.choice(CHANNELS, size=n, p=[0.28, 0.18, 0.14, 0.30, 0.10]),
        "billing_cycle": np.where(rng.random(n) < annual_p, "annual", "monthly"),
        "signup_idx": signup_idx,
        "signup_month": MONTHS[signup_idx],
        "initial_plan": initial_plan,
        "initial_seats": initial_seats,
    })
    df["signup_date"] = df["signup_month"] + pd.to_timedelta(rng.integers(0, 28, size=n), unit="D")
    df["updated_at"] = df["signup_date"]
    return df[df["signup_idx"] <= AS_OF_IDX]            # keep original index: it seeds each customer's RNG


# ------------------------------------------------------------------ subscriptions
def simulate_customer(c):
    rng = np.random.default_rng(SEED + 1 + c.Index)
    plan, seats = c.initial_plan, c.initial_seats
    annual = c.billing_cycle == "annual"
    rows = []
    for t, m in enumerate(range(c.signup_idx, AS_OF_IDX + 1)):
        u = rng.random(9)                               # always draw the same count -> stable history
        if t > 0:
            p = CHURN[plan] * (2.0 if t < 6 else 1.3 if t < 12 else 1.0)   # early-life churn is higher
            if annual:
                p = 1 - (1 - p) ** 12 if t % 12 == 0 else 0.0               # annual plans churn at renewal only
            if u[0] < p:
                break
            if u[1] < UPGRADE[plan]:
                plan = NEXT_PLAN[plan]
                seats = max(seats, SEAT_RANGE[plan][0])
            elif u[2] < DOWNGRADE[plan]:
                plan = PREV_PLAN[plan]
                seats = min(seats, SEAT_RANGE[plan][1])
            if u[3] < 0.06:                                                 # seat expansion
                seats = max(seats + 1, int(round(seats * (1.05 + 0.15 * u[4]))))
            elif u[3] > 0.98:                                               # seat contraction
                seats = max(1, min(seats - 1, int(seats * 0.9)))
        rows.append((c.customer_id, MONTHS[m], plan, seats,
                     seats * PRICE_PER_SEAT[plan], u[6], u[7], u[8]))
    return rows


def build_subscriptions(customers):
    rows = [r for c in customers.itertuples() for r in simulate_customer(c)]
    subs = pd.DataFrame(rows, columns=["customer_id", "month", "plan", "seats", "mrr",
                                       "u_pay1", "u_pay2", "u_pay3"])
    subs["tenure"] = subs.groupby("customer_id").cumcount()
    return subs


# ------------------------------------------------------------------ invoices
def build_invoices(subs, customers):
    seg = customers.set_index("customer_id")[["segment", "billing_cycle"]]
    df = subs.join(seg, on="customer_id")
    monthly = df[df["billing_cycle"] == "monthly"].copy()
    monthly["amount"] = monthly["mrr"]
    annual = df[(df["billing_cycle"] == "annual") & (df["tenure"] % 12 == 0)].copy()
    annual["amount"] = annual["mrr"] * 12               # billed up front at signup and each renewal

    inv = pd.concat([monthly, annual]).sort_values(["month", "customer_id"]).reset_index(drop=True)
    inv["invoice_id"] = [f"INV-{i + 1:07d}" for i in range(len(inv))]
    inv["invoice_date"] = inv["month"]
    inv["due_date"] = inv["invoice_date"] + pd.Timedelta(days=30)

    mean = inv["segment"].map({"SMB": 12, "Mid-Market": 28, "Enterprise": 42})
    sd = inv["segment"].map({"SMB": 6, "Mid-Market": 12, "Enterprise": 15})
    days = (mean + (inv["u_pay1"] - 0.5) * 2 * np.sqrt(3) * sd).round().clip(lower=1)
    days = np.where(inv["u_pay2"] < 0.04, days + 30 + (inv["u_pay3"] * 30).astype(int), days)  # ~4% pay very late
    paid = inv["invoice_date"] + pd.to_timedelta(days, unit="D")
    paid = paid.where(inv["u_pay2"] <= 0.985)           # ~1.5% never pay
    paid = paid.where(paid <= AS_OF_END)                # payments after the cut-off are not visible yet
    inv["paid_date"] = paid
    inv["updated_at"] = paid.fillna(inv["invoice_date"])  # watermark column: changes when payment lands
    inv["currency"] = "USD"
    return inv[["invoice_id", "customer_id", "invoice_date", "due_date", "amount",
                "currency", "paid_date", "updated_at"]]


# ------------------------------------------------------------------ marketing + financials
def build_marketing(customers):
    new = (customers.groupby(["signup_month", "acquisition_channel"]).size()
           .rename("new_customers").reset_index())
    rows = []
    for r in new.itertuples():
        seed = SEED + 10_000 + MONTHS.get_loc(r.signup_month) * 10 + CHANNELS.index(r.acquisition_channel)
        spend = r.new_customers * CAC[r.acquisition_channel] * np.random.default_rng(seed).uniform(0.85, 1.2)
        rows.append((r.signup_month, r.acquisition_channel, round(spend, 2)))
    mk = pd.DataFrame(rows, columns=["month", "channel", "spend"])
    mk["updated_at"] = mk["month"] + pd.offsets.MonthEnd(0)
    return mk


def build_financials(subs, marketing):
    rev = subs.groupby("month")["mrr"].sum()
    mk = marketing.groupby("month")["spend"].sum().reindex(rev.index, fill_value=0)
    out = []
    for m in rev.index:
        rng = np.random.default_rng(SEED + 20_000 + MONTHS.get_loc(m))
        r = rev[m]
        n = rng.uniform(0.97, 1.03, size=8)
        actual = {
            4000: r,
            4100: r * 0.04 * n[0],
            5000: r * 0.09 * n[1],
            5100: r * 0.06 * n[2],
            5200: r * 0.02 * n[3],
            6000: (120_000 + mk[m]) * n[4] + r * 0.04,
            6100: (180_000 + r * 0.10) * n[5],
            6200: (60_000 + r * 0.06) * n[6],
        }
        rev_bias = REV_BUDGET_BIAS[m.year] * rng.uniform(0.97, 1.03)
        for code, amt in actual.items():
            bias = rev_bias if code in (4000, 4100) else rng.uniform(0.95, 1.02)
            out.append((m, "Actual", code, round(amt, 2)))
            out.append((m, "Budget", code, round(amt * bias, 2)))
    fin = pd.DataFrame(out, columns=["month", "scenario", "account_code", "amount"])
    fin["updated_at"] = fin["month"] + pd.offsets.MonthEnd(0)
    return fin


# ------------------------------------------------------------------ main
def save(df, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(f"{OUT_DIR}/{name}.csv", index=False)
    print(f"{name:24s} {len(df):>9,} rows")


def main():
    customers = build_customers()
    subs = build_subscriptions(customers)
    invoices = build_invoices(subs, customers)
    marketing = build_marketing(customers)
    financials = build_financials(subs, marketing)

    # basic sanity checks before writing anything
    assert not subs.duplicated(["customer_id", "month"]).any(), "duplicate customer-month rows"
    assert (subs["mrr"] > 0).all(), "non-positive MRR"

    save(customers[["customer_id", "company_name", "segment", "region", "industry",
                    "acquisition_channel", "billing_cycle", "signup_date", "updated_at"]], "customers")
    subs_out = subs[["customer_id", "month", "plan", "seats", "mrr"]].copy()
    subs_out["updated_at"] = subs_out["month"] + pd.offsets.MonthEnd(0)
    save(subs_out, "subscriptions_monthly")
    save(invoices, "invoices")
    save(marketing, "marketing_spend")
    save(pd.DataFrame(ACCOUNTS, columns=["account_code", "account_name", "account_group"]), "chart_of_accounts")
    save(financials, "financials")

    now = subs[subs["month"] == pd.Timestamp(AS_OF)]
    print(f"\nAs of {AS_OF}: {len(customers):,} customers ever signed up, "
          f"{now['customer_id'].nunique():,} active, "
          f"MRR ${now['mrr'].sum():,.0f}, ARR ${now['mrr'].sum() * 12:,.0f}")


main()
