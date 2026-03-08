"""
scripts/generate_demo_inbox.py

Generates a realistic synthetic inbox of 847 emails for the Crasis demo.
No real PII. No API calls. Deterministic with --seed.

Output: demo/inbox_847.jsonl
Each line: {"id": int, "from": str, "subject": str, "body": str}

Usage:
    python scripts/generate_demo_inbox.py
    python scripts/generate_demo_inbox.py --seed 42 --out demo/inbox_847.jsonl
"""

import argparse
import json
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# Realistic-feeling data pools
# ---------------------------------------------------------------------------

URGENT_SENDERS = [
    ("Sarah Chen", "s.chen@acmecorp.com"),
    ("Marcus Webb", "mwebb@partners.io"),
    ("Priya Nair", "priya.nair@clientco.com"),
    ("David Okonkwo", "dokonkwo@legalgrp.com"),
    ("Rachel Torres", "r.torres@healthsys.org"),
    ("Tom Hargreaves", "thargreaves@fintech.co"),
    ("Ananya Sharma", "ananya@startup.ai"),
    ("James Whitfield", "jwhitfield@enterprise.com"),
    ("Mei-Ling Wu", "meiling@consult.net"),
    ("Carlos Reyes", "c.reyes@agency.co"),
]

WORK_SENDERS = [
    ("Alex Morgan", "alex@company.com"),
    ("Jordan Blake", "jblake@company.com"),
    ("Sam Patel", "s.patel@company.com"),
    ("Robin Osei", "rosei@company.com"),
    ("Casey Liu", "c.liu@company.com"),
    ("Dana Kowalski", "dkowalski@company.com"),
    ("Morgan Tran", "m.tran@company.com"),
    ("Riley Hassan", "rhassan@company.com"),
]

NEWSLETTER_SENDERS = [
    ("Morning Brew", "hello@morningbrew.com"),
    ("The Hustle", "news@thehustle.co"),
    ("HackerNews Digest", "digest@hndigest.com"),
    ("Product Hunt", "noreply@producthunt.com"),
    ("Substack Weekly", "weekly@substack.com"),
    ("TechCrunch Daily", "newsletter@techcrunch.com"),
    ("Bloomberg Tech", "tech@bloomberg.net"),
    ("Indie Hackers", "digest@indiehackers.com"),
    ("Pointer.io", "hi@pointer.io"),
    ("TLDR Newsletter", "dan@tldrnewsletter.com"),
]

RECEIPT_SENDERS = [
    ("Amazon", "auto-confirm@amazon.com"),
    ("Stripe", "receipts@stripe.com"),
    ("Notion", "billing@notion.so"),
    ("GitHub", "noreply@github.com"),
    ("Vercel", "billing@vercel.com"),
    ("Figma", "receipts@figma.com"),
    ("DigitalOcean", "noreply@digitalocean.com"),
    ("1Password", "receipts@1password.com"),
    ("Loom", "billing@loom.com"),
    ("Linear", "receipts@linear.app"),
]

SOCIAL_SENDERS = [
    ("LinkedIn", "notifications@linkedin.com"),
    ("Twitter/X", "notify@x.com"),
    ("GitHub", "noreply@github.com"),
    ("Slack", "feedback@slack.com"),
    ("Discord", "noreply@discord.com"),
    ("Notion", "team@notion.so"),
]

SPAM_SENDERS = [
    ("Win Big Today", "winner@prizes-now.biz"),
    ("Investment Alert", "alerts@stocksecrets.net"),
    ("Exclusive Offer", "deals@megasavings.info"),
    ("Your Package", "delivery@track-pkgs.co"),
    ("Account Notice", "security@verify-accounts.xyz"),
    ("Limited Time", "offers@flash-deals.biz"),
    ("Claim Reward", "rewards@claimfast.net"),
    ("Re: Your Inquiry", "noreply@enquiry-response.com"),
    ("Congratulations", "winner@lottery-intl.net"),
    ("Important Update", "update@acct-alert.info"),
]

# ---------------------------------------------------------------------------
# Template pools per category
# ---------------------------------------------------------------------------

URGENT_TEMPLATES = [
    (
        "URGENT: Contract review needed by EOD",
        "Hi, I need your eyes on the attached contract before 5pm today. The client is waiting and legal has flagged two clauses that need your approval. Can you get back to me ASAP?",
    ),
    (
        "Re: Production is down — need your call",
        "We've had an outage for 40 minutes. Payments are failing. I need you on a call right now. Ping me when you see this.",
    ),
    (
        "Time-sensitive: Board deck due tomorrow morning",
        "Just a reminder that the board deck needs to be finalized and sent tonight. I have slides 1-12, still waiting on your section. What's your ETA?",
    ),
    (
        "Action required: Wire transfer authorization",
        "The vendor is asking for authorization on the $48,000 wire before noon. Finance needs your sign-off. Please confirm you can do this today.",
    ),
    (
        "Please respond — interview candidate waiting",
        "We have a candidate who flew in for a final interview and we can't get the panel together. Are you available at 3pm or 4pm today? This person has another offer.",
    ),
    (
        "Critical bug in v2.3.1 — customer impacted",
        "Enterprise customer hit a data export bug. They're threatening escalation. We need a hotfix before their Monday 9am demo. Are you available this weekend?",
    ),
    (
        "Investor call moved up to tomorrow",
        "One of the Series A partners has a schedule conflict. They want to move the call to tomorrow at 10am EST. Please confirm asap so I can send the updated invite.",
    ),
    (
        "Re: Your quote — ready to sign today",
        "I've reviewed the proposal and we're ready to move forward. Can you get the agreement over today? My CFO is in the office until 4pm and can sign immediately.",
    ),
    (
        "Hospital system migration — go/no-go decision needed",
        "We're at the decision gate for the migration window. Your approval is the last thing blocking us. Go or no-go by 2pm or we push to next quarter.",
    ),
    (
        "Visa expiring — action required within 48 hours",
        "Your H-1B extension paperwork needs your wet signature. The attorney needs it by Thursday or we miss the filing window. Please call me today.",
    ),
]

WORK_TEMPLATES = [
    (
        "Q3 retrospective notes",
        "Hey, I finished compiling the retro notes from yesterday. Let me know if I missed anything or if you want me to send them to the broader team.",
    ),
    (
        "Quick question about the API docs",
        "I was reviewing the endpoint spec and noticed the rate limit for the /classify route seems inconsistent with what's in the README. Do you know which is correct?",
    ),
    (
        "Updated project timeline",
        "I revised the timeline based on our conversation. Milestone 2 is now pushed to the 18th. Let me know if that creates any conflicts on your end.",
    ),
    (
        "Intro: meet Jamie from the data team",
        "Connecting you two — Jamie is leading the new data infrastructure work and I think there's a good overlap with what you're building. Jamie, Adam is the person to talk to.",
    ),
    (
        "Next week's 1:1 — agenda?",
        "Do you have anything specific you want to cover in next week's 1:1? I have a few things on product direction and want to make sure we have time for your items too.",
    ),
    (
        "Design files are ready for review",
        "Uploaded the final mockups to Figma. Link in the thread. Main changes: new onboarding flow, updated dashboard layout, and the mobile nav. Lmk what you think.",
    ),
    (
        "Re: Pricing page copy",
        "I made the edits we discussed. The three-tier layout is looking clean. Still not sure about the CTA copy on the enterprise tier — open to suggestions.",
    ),
    (
        "Expenses from the NYC trip",
        "Submitting my expenses from the client visit. Total is $412. Receipts attached. Let me know if you need anything else from me for approval.",
    ),
]

NEWSLETTER_TEMPLATES = [
    (
        "Your morning briefing: 5 things to know today",
        "Good morning. Here's what's moving: AI funding hits $40B in Q1, remote work policies tighten at major banks, and a new study on sleep productivity is making the rounds.",
    ),
    (
        "🚀 This week in startups: the deals you missed",
        "Three companies raised over $100M this week. One of them is doing something genuinely interesting with local-first software. Here's the breakdown.",
    ),
    (
        "Top links from the community this week",
        "Members shared 847 links this week. These 5 got the most clicks: an essay on pricing strategy, a thread on LLM eval, and a breakdown of the new EU AI Act draft.",
    ),
    (
        "Product Hunt Daily: Today's top products",
        "Today's #1: A no-code tool for building internal apps. #2: An AI writing assistant focused on tone. #3: A Notion widget for habit tracking.",
    ),
    (
        "The algorithm changed again. Here's what matters.",
        "LinkedIn's latest update is deprioritizing link posts. Twitter engagement is down 12% YoY. Substack growth is up. Here's what the data actually says.",
    ),
    (
        "Weekly digest: best writing on the internet",
        "This week's picks: a long read on the future of programming, a short piece on why most dashboards are useless, and a thread that made engineers angry in the best way.",
    ),
]

RECEIPT_TEMPLATES = [
    (
        "Your receipt from Amazon — Order #112-8834421",
        "Thank you for your order. Items: USB-C hub (1x $34.99), mechanical keyboard switches (1x $18.50). Estimated delivery: Thursday. Track your package.",
    ),
    (
        "Invoice #INV-20240315 — $49.00 — Notion",
        "Your Notion Plus subscription has been renewed for another month. Amount charged: $49.00 to Visa ending in 4242. Questions? Reply to this email.",
    ),
    (
        "Payment confirmed: GitHub Copilot — $10.00",
        "Your GitHub Copilot subscription payment of $10.00 was processed successfully. Next billing date: April 15. Manage your subscription in Settings.",
    ),
    (
        "Stripe receipt — $2,400.00",
        "A payment of $2,400.00 was processed on your Stripe account. Description: Monthly platform fee from Acme Corp. Statement descriptor: ACME CORP.",
    ),
    (
        "Your Vercel invoice for March",
        "Invoice total: $20.00 (Pro plan). Usage within limits. No overages this month. Invoice available in your dashboard.",
    ),
    (
        "Figma Professional — renewal confirmation",
        "Your Figma Professional plan renewed at $15/month. Access all features including Dev Mode and unlimited projects. View invoice in account settings.",
    ),
]

SOCIAL_TEMPLATES = [
    (
        "You have 3 new connection requests on LinkedIn",
        "Sarah M., a Product Manager at Google, wants to connect. Also: two people from your industry who viewed your profile this week.",
    ),
    (
        "Your post is getting attention",
        "A post you shared 2 days ago has 847 impressions and 34 reactions. It's performing above average for your account.",
    ),
    (
        "New comment on your GitHub issue",
        "Someone commented on issue #142: 'This is blocking us too — would love to see this prioritized.' View the thread for context.",
    ),
    (
        "You were mentioned in a Slack message",
        "Jordan Blake mentioned you in #general: '@adam can you take a look at the deploy logs from this morning?' Click to open in Slack.",
    ),
    (
        "Discord: New message in your server",
        "You have 4 unread messages in the #builds channel. Last message: 'Anyone tested this on M1? Getting weird tokenizer behavior.'",
    ),
]

SPAM_TEMPLATES = [
    (
        "You've been selected — claim your $500 gift card",
        "Congratulations! As a valued customer, you've been randomly selected to receive a $500 Amazon gift card. Click here to claim before it expires in 24 hours.",
    ),
    (
        "Your account has been compromised — verify now",
        "Unusual activity detected on your account. To secure your account, verify your identity immediately. Click the link below before your account is suspended.",
    ),
    (
        "Make $5,000/week from home — no experience needed",
        "Our members are earning real money working just 2 hours a day from home. No experience required. Limited spots available. See how it works.",
    ),
    (
        "Re: Your package could not be delivered",
        "We attempted delivery of your package #PKG-2948374 but were unable to complete it. Reschedule delivery or your package will be returned. Action required.",
    ),
    (
        "Investment opportunity — 40% returns guaranteed",
        "Our proprietary algorithm has delivered 40%+ annual returns for 3 consecutive years. Spots in our Q2 fund close Friday. Request your investor briefing.",
    ),
    (
        "Hot penny stock alert: XBIO up 300% tomorrow?",
        "Our analysts have identified a catalyst event for XBIO that could drive 200-400% gains. Forward this to a friend. Past performance does not guarantee future results.",
    ),
    (
        "You qualify for a $250,000 business loan",
        "Based on your business profile, you pre-qualify for up to $250,000 in funding with no collateral required. Approval in 24 hours. Apply now.",
    ),
]

MISC_TEMPLATES = [
    (
        "Reminder: dentist appointment tomorrow at 2pm",
        "This is a reminder from Bright Smile Dental. Your appointment is scheduled for tomorrow, Tuesday, at 2:00 PM. Please arrive 10 minutes early.",
    ),
    (
        "Your flight itinerary: ORD → SFO",
        "Booking confirmation: United Airlines UA 234. Departure: ORD 7:45 AM, Arrival: SFO 10:22 AM. Seat: 22C. Confirmation: XKRT42.",
    ),
    (
        "Library book due in 3 days",
        "A reminder that 'The Pragmatic Programmer' is due back to the library on Thursday. Renew online or in person to avoid late fees.",
    ),
    (
        "Your Airbnb reservation is confirmed",
        "You're going to San Francisco! Check-in: March 14. Check-out: March 17. Host: Maria. Address sent 24 hours before check-in.",
    ),
    (
        "Gym membership renewal notice",
        "Your annual gym membership renews in 30 days. Current plan: Elite ($89/month). Log in to update payment or change your plan.",
    ),
    (
        "HOA meeting minutes — February",
        "Attached are the minutes from last month's HOA meeting. Key items: parking enforcement update, pool repair timeline, and budget approval for landscaping.",
    ),
    (
        "Re: Book club — next month's pick?",
        "Hey all, we need to vote on next month's book! Options so far: The Remains of the Day, Tomorrow and Tomorrow and Tomorrow, or Demon Copperhead. Reply with your vote.",
    ),
]


# ---------------------------------------------------------------------------
# Distribution (must sum to 847)
# ---------------------------------------------------------------------------

DISTRIBUTION = {
    "urgent": (URGENT_TEMPLATES, URGENT_SENDERS, 120),
    "work": (WORK_TEMPLATES, WORK_SENDERS, 185),
    "newsletter": (NEWSLETTER_TEMPLATES, NEWSLETTER_SENDERS, 210),
    "receipt": (RECEIPT_TEMPLATES, RECEIPT_SENDERS, 130),
    "social": (SOCIAL_TEMPLATES, SOCIAL_SENDERS, 95),
    "spam": (SPAM_TEMPLATES, SPAM_SENDERS, 75),
    "misc": (MISC_TEMPLATES, MISC_SENDERS := [
        ("Bright Smile Dental", "reminders@brightsmile.com"),
        ("United Airlines", "noreply@united.com"),
        ("Brooklyn Public Library", "notifications@bklynlibrary.org"),
        ("Airbnb", "automated@airbnb.com"),
        ("Equinox", "membership@equinox.com"),
        ("Riverside HOA", "admin@riversidehoa.org"),
        ("Book Club Group", "bookclub@gmail.com"),
    ], 32),
}

assert sum(v[2] for v in DISTRIBUTION.values()) == 847, \
    f"Distribution must sum to 847, got {sum(v[2] for v in DISTRIBUTION.values())}"


def generate_emails(seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    emails = []
    email_id = 1

    for category, (templates, senders, count) in DISTRIBUTION.items():
        for _ in range(count):
            subject, body = rng.choice(templates)
            name, address = rng.choice(senders)

            # Light variation: occasionally prefix subjects
            prefix_roll = rng.random()
            if prefix_roll < 0.08:
                subject = f"Re: {subject}"
            elif prefix_roll < 0.12:
                subject = f"Fwd: {subject}"

            emails.append({
                "id": email_id,
                "category": category,  # ground truth, not used by classifier
                "from_name": name,
                "from_address": address,
                "subject": subject,
                "body": body,
            })
            email_id += 1

    rng.shuffle(emails)

    # Re-assign IDs after shuffle so they're sequential
    for i, email in enumerate(emails, 1):
        email["id"] = i

    return emails


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic demo inbox")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--out",
        type=str,
        default="demo/inbox_847.jsonl",
        help="Output path",
    )
    parser.add_argument(
        "--no-category",
        action="store_true",
        help="Strip ground-truth category field from output",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating 847 emails (seed={args.seed})...")
    emails = generate_emails(seed=args.seed)

    if args.no_category:
        for e in emails:
            e.pop("category", None)

    with out_path.open("w") as f:
        for email in emails:
            f.write(json.dumps(email) + "\n")

    print(f"✓ Written to {out_path}")
    print(f"\nDistribution:")
    from collections import Counter
    counts = Counter(e["category"] for e in emails)
    for cat, n in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "█" * (n // 5)
        print(f"  {cat:<12} {n:>4}  {bar}")


if __name__ == "__main__":
    main()
