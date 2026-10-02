# Hallucinations

Mistakes the model made in live runs (Qwen3.8-27B on vLLM).

## 1. A limit the tool does not have

> **you>** What is the weather like in 8 Octomber?
>
> **bot>** I can't look up the weather for 8 October... the forecast tool only covers the next 24 hours, a specific day
> within the forecast range, or the current plan's time window. It doesn't provide forecasts that far out.

**Problem:** no tool was called. The model read the forecast tool's default (the next 24 hours when no argument is
given) as its range. Open-Meteo forecasts up to 16 days ahead.

## 2. Opening hours from nowhere

> **you>** Plan a historical walk in Corfu Town on Tuesday 6 October 2026 from 11:00 to 15:00.
>
> **bot>** Here is a walking itinerary for October 6, 2026, from 11:00 to 15:00, featuring five stops...
>
> **you>** What is the Liston?
>
> **bot>** The Liston is a famous covered promenade in Corfu Town, known for its arcades and elegant architecture. It is a
> popular outdoor spot for strolling and people-watching, and it is suitable for children. The Liston Arcades are open
> from 10:00 to 20:00 on weekdays, including Tuesday 6 October 2026.

**Problem:** no tool was called. The Liston was in the plan above, so the model answered from the conversation and its
own knowledge. The opening hours are invented. The dataset lists the Liston as open all day, every day.

## 3. A weather follow-up sent to the planner

> **you>** What is the weather like in 8 Octomber?
>
> **bot>** On Thursday 8 October 2026, Corfu will be mostly overcast with light rain and drizzle throughout the day...
>
> **you>** Ok tomorrow
>
> **bot>** I need a specific date along with a start and end time to build your Corfu itinerary. Please provide these
> details so I can check the curated dataset for you.

**State update:** the extraction read the message as a plan request for tomorrow with a forecast. No window was stored,
so none was set.

```python
plan_request = {"date": "2026-10-02", "sources": ["forecast"]}
window = None
```

**Problem:** the router sent "Ok tomorrow" to the plan path, which asked for a time window. The conversation was about
the weather.

## 4. An interest replaced instead of added

> **you>** We love food too.
>
> **bot>** I'm sorry, but I couldn't create a plan for your requested window because no stops fit the constraints.

**State update:** the stored preferences before and after the message.

```python
before = {"interests": ["history"], "with_child": True, "avoid_museums": False, "setting": None}
after = {"interests": ["food"], "with_child": True, "avoid_museums": False, "setting": None}
```

**Problem:** the stored interest was history. The extraction read "food" as the new list, not as an addition. With a
child in the group, no food place is suitable for children, so every stop was removed.

## 5. A typo refused

> **you>** What museums are available?
>
> **bot>** Corfu has several museums you can visit...
>
> **you>** What is the ticker price?
>
> **bot>** I can help with visiting Corfu: questions about its places, recommendations, half-day plans and the weather.

**Problem:** the router read "ticker" as a question outside Corfu and refused it. "What is the ticket price?" gets an
answer.

## 6. An outdoor setting nobody asked for

> **you>** Plan a four-hour walking itinerary in Corfu focused on historical attractions on Tuesday, October 6, 2026,
> from 11:00 to 15:00.

**State update:** the extraction wrote a setting the message does not state.

```python
plan_request = {
    "date": "2026-10-06",
    "start_time": "11:00",
    "end_time": "15:00",
    "interests": ["history"],
    "setting": "outdoor",
}
```

**Problem:** "walking" was read as "outdoor".
