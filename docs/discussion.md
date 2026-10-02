# Discussion

## Retrieval quality

Each place is one document (its name, category, tags, area and description), embedded with MiniLM and searched by
cosine similarity, top 4.

| Question | Top score | 4th score |
|---|---|---|
| Who built the Old Fortress and when? | 0.58 | 0.51 |
| Venetian fortifications | 0.64 | 0.48 |
| A museum with Asian art | 0.71 | 0.39 |
| An aquarium in Corfu Town (not in the dataset) | 0.51 | 0.45 |
| Rent a jet ski (not in the dataset) | 0.23 | 0.19 |
| The best pizza (not in the dataset) | 0.38 | 0.36 |

- A cutoff around 0.45 drops the clear misses, but not the aquarium (MiniLM links it to the waterfront).
- So the prompt's grounding rule decides "not in the dataset", not a threshold.
- Recommendations filter on metadata first, and similarity only orders what qualifies.
- In the [evaluation](evaluation.md), every search found its record (the Old Fortress, Saint Spyridon).
- `find_places` returned five places for "Venetian history", each fact from its cited record.

## The iterative design process

All three approaches ran on the [evaluation](evaluation.md)'s conversations, with Qwen3.8-27B on vLLM.

### Approach 1. One wide schema

Every message was read into one form covering every intent, then handled by that intent.

```
ParsedRequest
  intent:            question | recommend | plan | edit_plan | weather | out_of_scope
  question:          the question, for intent=question
  date, start_time, end_time
  horizon_hours:     for "the next N hours"
  interests:         [free-text topics]
  requested_places:  [place names]
  stop_index:        the stop to change
  with_child:        true | false | null
  avoid_museums:     true | false | null
  setting:           indoor | outdoor | null
  wants_fabrication: true | false
```

**Why it brings hallucinations**

- No "not stated" value and free-text interests, so the model fills values never stated (the five named places, Venetian history).
- An invented value reshapes the plan. The outdoor history plan became one Achilleion stop, so the rain had nothing to replace.
- Reply and stored plan drift apart. In the visitor's narrative it said every stop was removed but kept them.

### Approach 2. One agent for everything

- ✔️ With its tools, it built the same plans in every conversation and refused the attack.
- ❌ It may skip tools. In the visitor's narrative it claimed a plan update and a forecast it never fetched.
- ❌ <ins>Costly. Schemas, prompt and history go into every call, about 2× the router's and 4× the wide schema's tokens.</ins>

### Approach 3. The middle ground, a [custom router workflow](https://docs.langchain.com/oss/python/langchain/multi-agent/custom-workflow) (agentic vs. deterministic)

An architecture that mixes deterministic logic with agentic behavior. The route node sends plan messages to code and
questions to one agent.

```
route      Route         path: plan | ask | attack | out_of_scope
─────────────────────────────────────────────────────────────────────────────────────────────────────────
extract    PlanRequest   date, start_time, end_time: a value or null
                         interests:        [history | venetian | museums | art | nature | food | architecture]
                         child, avoid_museums, keep_other_stops: yes | no | not_stated
                         setting:          indoor | outdoor | any | not_stated
                         places:           [place names]
                         replace_stops:    [stop numbers]
                         sources:          [forecast]
─────────────────────────────────────────────────────────────────────────────────────────────────────────
ask        tools         search(query)
                         find_places(interests, child, avoid_museums, setting, date)
                         forecast(stop | visit_window | date | next_hours)
─────────────────────────────────────────────────────────────────────────────────────────────────────────
respond    free text     2-5 framing sentences
```

**Why it works** (isolation of concerns per node, with fewer responsibilities, less schema payload overhead and fewer hallucinations)

- Code checks the hard constraints, and the model only reads requests and words replies (every plan conversation).
- Follow-ups change only what was asked. The history walk replaced stop 2, the rain replanning the stop in the rain.
- Cheap refusals. An attack costs one route call and a fixed reply, 0.3 s in the evaluation, with the plan untouched.
- <ins>Trade-off. One extra route call, at about half the agent's tokens, with citations limited to the records used.</ins>

## Hallucinations

Mistakes seen in live runs.

1. **A limit the tool does not have.**
   - "What is the weather like on 8 October?" got "the forecast tool only covers the next 24 hours".
   - No tool was called, and the model read the tool's default as its range.
   - Open-Meteo forecasts 16 days.

   The prompt now says never to decide a tool cannot answer without calling it.

2. **Answers from the conversation.**
   - A question about a place already in the plan is often answered with no tool call.
   - Mostly harmless when the record was cited earlier, and even an efficiency gain with no extra tool call.
   - In the [evaluation](evaluation.md#2-historical-walk-and-a-replaced-stop---vllmqwen38-27b), the Ionian
     Academy got "open courtyards and grand halls", while its record describes a short exterior stop.

   The prompt asks for this turn's tool results only, and no code checks it.

3. **A typo refused.**
   - "What is the ticker price?" was refused as off-topic. "Ticket" gets an answer.

   Rephrasing gets an answer.

4. **An outdoor setting nobody asked for.**
   - "A four-hour walking itinerary" was read as an outdoor setting.

   The extraction prompt now says walking means on foot, not outdoor.

5. **A date nobody gave.**
   - "Start from 11am" filled in today's date, and the plan was built for today without asking.

   The code asks for a missing date, but cannot tell one the model filled in.

6. **A kind of place read as a name.**
   - "Replace a stop for a restaurant" set no interest, and a subsequent turn put "restaurant" in the named places.
   - The reply said the dataset has no restaurant. It has nine food places.

   The prompt now maps a restaurant, taverna, café or meal to food, and keeps kinds of place out of names.

> General applied fixes were stricter prompts and the continuous refinement of the state, with small schemas that hold
> an explicit "not stated", closed lists and a fixed reply for refusals. Models can still make mistakes and fill wrong
> values, but each value is validated against the schema and each plan against each hard constraint of the domain
> layer. Every change is made explicit in the run through the state, logging, and corresponding traces.
