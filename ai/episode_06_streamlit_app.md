# Episode 6 — Streamlit Investor Journey

## Task

I needed to turn the project outputs into an investor-facing Streamlit prototype.

The application needed to support a realistic customer journey rather than simply display static charts.

## AI Use

AI helped structure the Streamlit code, debug local execution issues and improve the layout of the four main tabs.

## Final App Structure

### Compare Funds

Allows the user to compare all twelve funds using:

- CAGR
- volatility
- Sharpe
- maximum drawdown
- return-versus-risk visualisation

### Fund Fact Sheet

Allows the user to inspect an individual fund using:

- performance metrics
- growth of $1
- latest target holdings
- concentration warning

### Build an Allocation

Allows the user to:

- select up to four funds,
- assign portfolio percentages,
- require the allocation to sum to 100%,
- calculate blended historical performance from the precomputed fund return series.

The allocation is calculated dynamically rather than displaying a fixed example.

### Sentiment Analytics

Allows the user to compare:

- Standard VADER
- Finance-Adjusted VADER
- neutral rates
- finance-term coverage
- sector sentiment through time
- news coverage

## What I Checked

I ran the application locally using:

`python -m streamlit run streamlit_app.py`

I tested each tab individually and checked that:

- the app loads the saved project artifacts,
- it does not rerun the full backtest,
- it does not run VADER during page load,
- allocation percentages must sum to 100%,
- the sentiment figures and metrics match the project outputs,
- concentration information uses the latest available backtest target weights rather than claiming they are live holdings.

## Corrections

During testing I also corrected presentation issues such as:

- deprecated Streamlit width arguments,
- percentage versus percentage-point labelling,
- fact-sheet terminology,
- app wording that could imply the prototype was executing real trades.

## Final Outcome

The locally tested Streamlit app supports the full investor journey required by the project while using precomputed outputs to remain lightweight.

## Reflection

AI helped accelerate the interface build, but the app still required repeated local testing because UI code that compiles successfully does not guarantee that the user journey works correctly.
