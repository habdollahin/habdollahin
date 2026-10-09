# Vibe coding example: a tiny DVH tool

A worked example of using vibe coding to *learn*, not just to get code.
The topic is a dose-volume histogram (DVH), the same idea as `dvh_metrics` in
`dosevlm/metrics.py`, rewritten in plain Python so every line is readable.

```
python3 dvh_tool.py                    # print a DVH report for a fake PTV and lung
python3 -m unittest test_dvh_tool.py   # run the tests
```

## How it was built: the rounds

**Round 1 – describe the goal.**
> "Write a Python tool that takes a list of voxel doses for a structure and prints
> the mean dose, max dose, D95 and V20. No libraries, keep it beginner-friendly."

Result: `mean_dose`, `dose_at_volume` (D_x), `volume_at_dose` (V_x).

**Round 2 – add a feature by describing what you see.**
> "Also show the DVH curve as a text bar chart in the terminal."

Result: `dvh_curve` and the `#####` bars in `report`.

**Round 3 – break it and fix it.**
> "What happens if a structure is empty?"

It crashed with `ZeroDivisionError`. The fix is the `_check` guard, which raises a
clear `ValueError` instead. Lesson: "it works when I run it" is not enough;
edge cases have to be tried on purpose.

**Round 4 – lock it in with tests.**
> "Write tests with simple numbers I can check by hand."

Result: `test_dvh_tool.py`. Doses 10…100 Gy make every answer easy to verify
in your head (mean = 55, D50 = 60, V55 = 50%).

## Your turn: exercises

Do these yourself first. Ask Claude for a **hint**, not the answer, if you get stuck.

1. **Read.** In `dose_at_volume`, why is there `max(1, ...)`? What breaks without it?
   (Try `dose_at_volume([50, 60], 1)`.)
2. **Predict, then run.** What will `volume_at_dose([10, 20, 30], 20)` return? Check it.
3. **Add a feature.** Write `homogeneity_index(doses)` = (D2 − D98) / D50.
   Add a test for it.
4. **Add a feature.** Make `report` also print D2 (near-max dose) for targets.
5. **Compare.** Open `dosevlm/metrics.py` and find `dvh_metrics`. How does the
   NumPy version compute D95? Do both versions give the same number on the PTV data?
6. **Review.** Paste your code back and ask: "Review this like a senior developer.
   What's wrong, what's good, and why?"
