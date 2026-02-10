# Requirements Document

## Introduction

This document specifies requirements for a Wikipedia Product Health Analysis system that evaluates Wikipedia's product health using time-series data from 2015-2025. The system directly addresses reviewer concerns about "naive traffic analysis" by mandating rigorous statistical validation, causal inference, and multi-dimensional analysis for every conclusion.

**Key Differentiators from Descriptive EDA:**

- Every trend claim requires statistical significance testing (p-values, confidence intervals)
- Every causal claim requires causal inference methodology (interrupted time series, difference-in-differences, synthetic controls)
- Every conclusion requires cross-validation across multiple data sources (pageviews, editors, edit volume)
- Every forecast includes uncertainty quantification (prediction intervals, scenario analysis)

This system transforms raw traffic data into evidence-backed product insights through a multi-layered validation framework that ensures no conclusion lacks supporting statistical arguments.

**Research Questions Addressed:**

This analysis answers 11 specific research questions about Wikipedia's product health:

1. **AI-Assisted Search Impact**: Structural changes in usage trends coinciding with AI search rise (ChatGPT Nov 2022, Google AI Overviews May 2024)
2. **Future Traffic Projections**: Wikipedia's future traffic if current patterns continue
3. **Mobile Dependency Risk**: Over-reliance on mobile traffic and associated risks
4. **Usage Pattern Changes**: Altered usage patterns during specific time periods and growth rate evolution
5. **Mobile App vs Mobile Web Stability**: Engagement stability differences between mobile platforms
6. **Campaign Effectiveness**: Measurable short-term and long-term traffic changes from campaigns
7. **Mobile App Product Health**: Product health metrics specific to Wikipedia's mobile app
8. **Weekday vs Weekend Usage**: Usage patterns suggesting utility vs leisure behavior
9. **Platform Growth Drivers**: Which platforms drive overall growth or decline
10. **Long-term Reliance Trends**: Changing user reliance on Wikipedia over time
11. **Global Disruption Response**: Wikipedia usage response to major news cycles and disruptions

## Glossary

- **Analysis_System**: The complete Wikipedia product health analysis platform
- **Data_Acquisition_Module**: Component responsible for fetching data from Wikimedia APIs
- **Statistical_Validator**: Component that performs hypothesis testing and significance analysis
- **Causal_Analyzer**: Component that performs causal inference and attribution analysis
- **Time_Series_Engine**: Component that handles temporal pattern analysis and forecasting
- **Evidence_Framework**: The integrated system for validating conclusions with statistical rigor
- **Multi_Dimensional_Analyzer**: Component that correlates pageviews with engagement, editor activity, and content quality
- **Pageview**: A single request for a Wikipedia page, filtered to exclude bot traffic (agent=user)
- **Active_Editor**: A user who made at least one edit in a given time period
- **Edit_Volume**: Total number of edits made across Wikipedia in a time period
- **Engagement_Metric**: Measures of user interaction depth (session duration, pages per session, bounce rate)
- **Campaign**: A deliberate intervention or initiative by Wikipedia (e.g., fundraising, awareness)
- **External_Shock**: Significant external events that may impact Wikipedia usage (e.g., news events, global crises)
- **Structural_Shift**: A statistically significant change in long-term usage patterns detected by changepoint algorithms
- **Effect_Size**: The magnitude of an observed impact, quantified with confidence intervals (e.g., Cohen's d, percentage change)
- **Confidence_Interval**: Statistical range indicating uncertainty in estimates (typically 95%)
- **P_Value**: Probability that observed results occurred by chance under the null hypothesis
- **Significance_Level**: Threshold for statistical significance (α = 0.05 unless otherwise specified)
- **Counterfactual**: Predicted outcome in the absence of an intervention, used for causal inference
- **Synthetic_Control**: Weighted combination of control units used to construct counterfactual baselines
- **Treatment_Effect**: Causal impact of an intervention, measured as difference between observed and counterfactual outcomes
- **Wikimedia_REST_API**: Primary API for accessing pageview data (https://wikimedia.org/api/rest_v1/)
- **Wikimedia_Analytics_API**: API for accessing editor and edit metrics
- **Access_Method**: Platform type - desktop, mobile-web, or mobile-app
- **Agent_Type**: User type filter - user (human) or spider (bot)
- **Granularity**: Time aggregation level - hourly, daily, or monthly
- **Project**: Wikipedia project identifier (e.g., en.wikipedia for English Wikipedia)

## Requirements

### Requirement 1: Data Acquisition from Wikimedia APIs

**User Story:** As a data analyst, I want to acquire comprehensive time-series data from Wikimedia APIs with specific endpoints and parameters, so that I can perform rigorous multi-source validation of findings.

**Data Sources Specified:**

- **Pageviews API**: `https://wikimedia.org/api/rest_v1/metrics/pageviews/aggregate/{project}/{access}/{agent}/{granularity}/{start}/{end}`
- **Top Pages API**: `https://wikimedia.org/api/rest_v1/metrics/pageviews/top/{project}/{access}/{year}/{month}/{day}`
- **Editors API**: Via Wikimedia Analytics datasets or XTools API
- **Edit Volume API**: Via Wikimedia Analytics datasets

**Research Questions Addressed:** All questions (1-11) require this foundational data

#### Acceptance Criteria

1. WHEN requesting pageview data for AI impact analysis (Q1), THE Data_Acquisition_Module SHALL fetch monthly aggregated data from 2019-01-01 to present using endpoint `/metrics/pageviews/aggregate/en.wikipedia/{access}/user/monthly/{start}/{end}` for each access method (desktop, mobile-web, mobile-app)

2. WHEN fetching pageview data, THE Data_Acquisition_Module SHALL set agent=user to exclude bot traffic and ensure only human pageviews are analyzed

3. WHEN acquiring platform-specific data (Q3, Q5, Q9), THE Data_Acquisition_Module SHALL make separate API calls for access=desktop, access=mobile-web, and access=mobile-app to enable platform comparison

4. WHEN gathering editor metrics for engagement analysis (Q4, Q7, Q10), THE Data_Acquisition_Module SHALL acquire active editor counts from Wikimedia Analytics datasets with monthly granularity

5. WHEN collecting edit volume data for cross-validation (Q4, Q10), THE Data_Acquisition_Module SHALL acquire edit counts from Wikimedia Analytics with daily or monthly granularity matching pageview data

6. WHEN requesting data for weekday analysis (Q8), THE Data_Acquisition_Module SHALL fetch daily granularity pageviews using `/metrics/pageviews/aggregate/en.wikipedia/all-access/user/daily/{start}/{end}` to enable day-of-week aggregation

7. WHEN data acquisition completes, THE Data_Acquisition_Module SHALL validate data completeness by checking for gaps in date ranges and flag any missing dates or anomalies in a validation report

8. WHEN storing acquired data, THE Data_Acquisition_Module SHALL persist data in structured format with metadata including: API endpoint used, query parameters, acquisition timestamp, date range, access method, agent type, and granularity

### Requirement 2: AI-Assisted Search Impact Analysis (Research Question 1)

**User Story:** As a product strategist, I want to detect and quantify structural changes in Wikipedia usage coinciding with AI search adoption (ChatGPT launch Nov 2022, Google AI Overviews May 2024), so that I can understand AI's impact on Wikipedia traffic with statistical evidence.

**Specific Analysis Methods:**

- **Data Range**: Monthly pageviews from 2019-01-01 to present (captures 3+ years pre-ChatGPT baseline)
- **Changepoint Detection**: Apply PELT, Binary Segmentation, and Bayesian changepoint algorithms to detect structural breaks
- **Key Dates to Test**: November 2022 (ChatGPT launch), May 2024 (Google AI Overviews rollout)
- **Platform Comparison**: Analyze desktop vs mobile separately (hypothesis: AI search more prevalent on desktop)
- **Statistical Tests**: Chow test for structural breaks, interrupted time series analysis with intervention dates

#### Acceptance Criteria

1. WHEN analyzing AI search impact, THE Time_Series_Engine SHALL fetch monthly pageview data from 2019-01-01 to present for desktop, mobile-web, and mobile-app separately using the Pageviews API with monthly granularity

2. WHEN detecting structural breaks, THE Time_Series_Engine SHALL apply at least two changepoint detection algorithms (PELT and Bayesian) and identify consensus changepoints where algorithms agree within ±2 months

3. WHEN a changepoint is detected, THE Statistical_Validator SHALL perform a Chow test to confirm the structural break is statistically significant at α = 0.05

4. WHEN evaluating temporal alignment with AI events, THE Causal_Analyzer SHALL test whether detected changepoints align with November 2022 (±3 months) or May 2024 (±3 months) and calculate the probability of such alignment occurring by chance

5. WHEN comparing platforms, THE Statistical_Validator SHALL calculate pre-break and post-break growth rates for desktop vs mobile with 95% confidence intervals to test if desktop shows larger decline

6. WHEN quantifying impact magnitude, THE Analysis_System SHALL calculate percentage change in traffic levels pre-break vs post-break with confidence intervals and effect sizes (Cohen's d)

7. WHEN cross-validating findings, THE Multi_Dimensional_Analyzer SHALL correlate pageview changes with editor activity changes to distinguish between reduced passive consumption vs reduced active engagement

8. WHEN reporting AI impact findings, THE Analysis_System SHALL provide: (a) changepoint dates with confidence intervals, (b) Chow test statistics and p-values, (c) pre/post growth rate comparison, (d) temporal alignment analysis with AI event dates, (e) platform-specific impact quantification, (f) editor activity correlation results

### Requirement 3: Future Traffic Projections (Research Question 2)

**User Story:** As a capacity planner, I want probabilistic traffic forecasts with multiple scenarios and uncertainty quantification, so that I can plan infrastructure and understand best/worst case outcomes.

**Specific Analysis Methods:**

- **Forecast Horizon**: 12-24 months ahead
- **Models**: ARIMA (captures autocorrelation), Prophet (handles seasonality and holidays), Exponential Smoothing (baseline), ensemble average
- **Scenarios**: Optimistic (trend reversal), baseline (trend continuation), pessimistic (accelerated decline)
- **Uncertainty**: 50%, 80%, 95% prediction intervals

#### Acceptance Criteria

1. WHEN generating traffic forecasts, THE Time_Series_Engine SHALL implement three forecasting methods: ARIMA with auto-selected order, Prophet with multiplicative seasonality, and Exponential Smoothing with seasonal periods=12 (monthly data)

2. WHEN producing ensemble forecasts, THE Time_Series_Engine SHALL average predictions from all three models weighted by their cross-validation performance (inverse MAPE weighting)

3. WHEN quantifying uncertainty, THE Time_Series_Engine SHALL provide 50%, 80%, and 95% prediction intervals for all forecasts using each model's native uncertainty estimation

4. WHEN evaluating forecast accuracy, THE Statistical_Validator SHALL perform cross-validation on the most recent 12 months as holdout data and compute MAPE, RMSE, MAE, and MASE for each model

5. WHEN comparing models, THE Statistical_Validator SHALL perform Diebold-Mariano tests to determine if accuracy differences between models are statistically significant

6. WHEN generating scenarios, THE Time_Series_Engine SHALL create: (a) optimistic scenario assuming trend reversal to historical growth rates, (b) baseline scenario continuing current trends, (c) pessimistic scenario assuming 1.5x current decline rate

7. WHEN incorporating external factors, THE Time_Series_Engine SHALL include known future events (e.g., major anniversaries, planned campaigns) as covariates in Prophet model

8. WHEN reporting forecasts, THE Analysis_System SHALL present: (a) point forecasts with uncertainty fans, (b) model comparison table with accuracy metrics and DM test p-values, (c) scenario analysis with probability-weighted outcomes, (d) key assumptions and limitations, (e) recommended forecast update frequency

### Requirement 4: Mobile Dependency Risk Assessment (Research Question 3)

**User Story:** As a product strategist, I want to quantify mobile dependency risks with measurable thresholds and concentration metrics, so that I can prioritize platform-specific initiatives based on concrete risk data.

**Specific Analysis Methods:**

- **Platform Mix Calculation**: Proportion of traffic from desktop, mobile-web, mobile-app
- **Risk Threshold**: 70% mobile dependency (industry benchmark for high concentration risk)
- **Concentration Metric**: Herfindahl-Hirschman Index (HHI) where HHI > 2500 indicates high concentration
- **Volatility Comparison**: Coefficient of variation for each platform
- **Scenario Analysis**: Impact of 10%, 20%, 30% decline in dominant platform

#### Acceptance Criteria

1. WHEN calculating platform mix, THE Analysis_System SHALL compute the proportion of total pageviews from desktop, mobile-web, and mobile-app for each month with 95% confidence intervals using bootstrap resampling (10,000 iterations)

2. WHEN assessing platform trends, THE Time_Series_Engine SHALL calculate compound annual growth rate (CAGR) for each platform over the full analysis period (2015-2025) with confidence intervals

3. WHEN evaluating mobile dependency risk, THE Statistical_Validator SHALL test whether combined mobile traffic (mobile-web + mobile-app) exceeds 70% threshold and calculate the probability of crossing 75% and 80% thresholds within 12 months using forecast distributions

4. WHEN computing concentration risk, THE Analysis_System SHALL calculate Herfindahl-Hirschman Index as HHI = Σ(platform_share²) × 10,000 and classify risk as: low (HHI < 1500), moderate (1500-2500), high (> 2500)

5. WHEN comparing platform stability, THE Analysis_System SHALL calculate coefficient of variation (CV = std/mean) for each platform's monthly pageviews and test whether CVs differ significantly using Levene's test for equality of variances

6. WHEN analyzing platform switching behavior, THE Causal_Analyzer SHALL use difference-in-differences to compare platform-specific responses to external events (e.g., mobile app updates, desktop redesigns) to identify platform-specific sensitivities

7. WHEN performing scenario analysis, THE Analysis_System SHALL model the impact on total traffic of 10%, 20%, and 30% declines in the dominant platform, assuming other platforms remain constant, and calculate resulting total traffic with confidence intervals

8. WHEN reporting platform risks, THE Analysis_System SHALL provide: (a) current platform mix with confidence intervals and trend direction, (b) CAGR for each platform with significance tests, (c) HHI score with risk classification, (d) volatility comparison with Levene's test results, (e) scenario analysis showing traffic impact of dominant platform decline, (f) recommendations for platform diversification

### Requirement 5: Usage Pattern Evolution Analysis (Research Question 4)

**User Story:** As a product analyst, I want to identify how Wikipedia usage patterns have changed during specific time periods and understand growth rate evolution, so that I can detect shifts in user behavior and product-market fit.

**Specific Analysis Methods:**

- **Time Period Segmentation**: Pre-pandemic (2015-2019), pandemic (2020-2021), post-pandemic (2022-present)
- **Growth Rate Analysis**: Calculate rolling 12-month growth rates to detect acceleration/deceleration
- **Pattern Detection**: Seasonal decomposition to separate trend from seasonality
- **Engagement Correlation**: Ratio of editors to pageviews as engagement quality metric

#### Acceptance Criteria

1. WHEN segmenting time periods, THE Analysis_System SHALL divide the data into: pre-pandemic (2015-01-01 to 2019-12-31), pandemic (2020-01-01 to 2021-12-31), post-pandemic (2022-01-01 to present), and AI-era (2023-01-01 to present)

2. WHEN calculating growth rates, THE Time_Series_Engine SHALL compute rolling 12-month year-over-year growth rates for each month and identify periods of acceleration (growth rate increasing) vs deceleration (growth rate decreasing)

3. WHEN comparing time periods, THE Statistical_Validator SHALL perform ANOVA to test whether mean pageviews differ significantly across the four time periods and compute effect sizes (eta-squared) for the magnitude of differences

4. WHEN analyzing pattern changes, THE Time_Series_Engine SHALL perform seasonal decomposition (STL) for each time period separately and test whether seasonal amplitude has changed significantly using F-tests

5. WHEN evaluating engagement quality, THE Multi_Dimensional_Analyzer SHALL calculate the ratio of active editors per 1 million pageviews for each time period and test whether this ratio has changed significantly using t-tests

6. WHEN detecting behavioral shifts, THE Analysis_System SHALL identify the top 10 most-viewed articles in each time period and categorize them (news, reference, entertainment, education) to detect shifts in content consumption patterns

7. WHEN cross-validating findings, THE Evidence_Framework SHALL confirm that pageview pattern changes align with edit volume pattern changes, ensuring consistency across engagement metrics

8. WHEN reporting usage evolution, THE Analysis_System SHALL provide: (a) time period comparison with ANOVA results, (b) growth rate trajectory plot with acceleration/deceleration phases marked, (c) seasonal pattern evolution analysis, (d) engagement ratio trends with significance tests, (e) content consumption pattern shifts, (f) cross-validation results across metrics

### Requirement 6: Mobile App vs Mobile Web Stability Comparison (Research Question 5)

**User Story:** As a mobile product manager, I want to compare engagement stability between mobile app and mobile web, so that I can understand which platform provides more consistent user engagement and prioritize platform investments.

**Specific Analysis Methods:**

- **Stability Metrics**: Coefficient of variation, volatility (standard deviation of returns), autocorrelation
- **Engagement Depth**: Session metrics if available, or use editors-per-pageview as proxy
- **Retention Proxy**: Month-over-month correlation (high correlation = stable user base)
- **Growth Comparison**: CAGR and trend consistency

#### Acceptance Criteria

1. WHEN comparing platform stability, THE Analysis_System SHALL calculate coefficient of variation (CV) for mobile-app and mobile-web monthly pageviews separately and test whether CVs differ significantly using Levene's test or F-test

2. WHEN measuring volatility, THE Time_Series_Engine SHALL calculate month-over-month percentage changes for each platform and compute standard deviation of these changes as volatility metric, then test for significant differences using variance ratio tests

3. WHEN assessing consistency, THE Statistical_Validator SHALL compute autocorrelation at lag-1 for each platform's time series (high autocorrelation indicates stable, predictable patterns) and test whether autocorrelations differ significantly

4. WHEN evaluating engagement depth, THE Multi_Dimensional_Analyzer SHALL calculate editors-per-million-pageviews ratio for mobile-app vs mobile-web and test whether engagement depth differs significantly between platforms using t-tests

5. WHEN analyzing growth patterns, THE Time_Series_Engine SHALL fit linear trends to each platform's log-transformed pageviews and compare trend slopes and R² values to assess growth consistency

6. WHEN testing retention proxies, THE Analysis_System SHALL calculate rolling 3-month correlation between consecutive periods for each platform (high correlation suggests stable returning user base) and compare correlation distributions

7. WHEN performing robustness checks, THE Evidence_Framework SHALL repeat stability analysis with outliers removed and confirm that stability rankings remain consistent

8. WHEN reporting stability comparison, THE Analysis_System SHALL provide: (a) CV comparison with significance tests, (b) volatility metrics with variance ratio test results, (c) autocorrelation comparison, (d) engagement depth comparison, (e) growth consistency analysis, (f) retention proxy results, (g) recommendation on which platform shows more stable engagement

### Requirement 7: Campaign Effectiveness Measurement (Research Question 6)

**User Story:** As a marketing analyst, I want to measure short-term and long-term traffic impacts of Wikipedia campaigns with causal inference methods, so that I can determine true incremental effects and optimize future campaign investments.

**Specific Analysis Methods:**

- **Causal Method**: Interrupted Time Series Analysis (ITSA) with segmented regression
- **Counterfactual**: Synthetic control or ARIMA forecast of baseline without campaign
- **Time Windows**: Immediate (0-7 days), short-term (8-30 days), long-term (31-90 days)
- **Significance Testing**: Permutation tests with 1000+ permutations
- **Effect Metrics**: Average Treatment Effect (ATE), cumulative effect, percentage lift

#### Acceptance Criteria

1. WHEN identifying campaigns, THE Analysis_System SHALL compile a list of major Wikipedia campaigns with dates from 2015-2025 including: annual fundraising campaigns (typically November-December), Wikipedia birthday campaigns (January 15), and major awareness initiatives

2. WHEN evaluating a campaign, THE Causal_Analyzer SHALL implement interrupted time series analysis with segmented regression modeling: level change (immediate effect) and slope change (sustained effect) at the campaign start date

3. WHEN constructing counterfactuals, THE Causal_Analyzer SHALL build synthetic control baselines using 90 days of pre-campaign data, fitting ARIMA or Prophet models to forecast expected traffic without the campaign, with 95% prediction intervals

4. WHEN measuring immediate effects, THE Statistical_Validator SHALL calculate average treatment effect for days 0-7 post-campaign as mean(observed - predicted) with 95% confidence intervals using bootstrap (10,000 iterations)

5. WHEN assessing short-term effects, THE Statistical_Validator SHALL calculate ATE for days 8-30 post-campaign and test whether it differs significantly from zero using permutation tests (minimum 1000 permutations)

6. WHEN evaluating long-term effects, THE Statistical_Validator SHALL calculate ATE for days 31-90 post-campaign and measure effect decay using exponential decay model to estimate half-life

7. WHEN comparing campaigns, THE Statistical_Validator SHALL test whether effect sizes differ significantly across campaigns using meta-analysis or hierarchical modeling, controlling for campaign duration and intensity

8. WHEN cross-validating campaign effects, THE Multi_Dimensional_Analyzer SHALL check whether pageview increases align with editor activity increases (true engagement) or remain isolated to pageviews (passive consumption spike)

9. WHEN performing robustness checks, THE Causal_Analyzer SHALL conduct placebo tests by applying the same analysis to non-campaign periods and verify that placebo effects are not significant

10. WHEN reporting campaign effectiveness, THE Analysis_System SHALL provide: (a) observed vs counterfactual time series plots with confidence bands, (b) ATE for each time window with confidence intervals and p-values, (c) cumulative effect (total incremental pageviews), (d) percentage lift over baseline, (e) effect duration and decay analysis, (f) cross-campaign comparison with meta-analysis results, (g) engagement quality assessment, (h) cost-effectiveness metrics if budget data available

### Requirement 8: Mobile App Product Health Assessment (Research Question 7)

**User Story:** As a mobile product manager, I want comprehensive product health metrics for Wikipedia's mobile app, so that I can assess app performance, identify issues, and prioritize improvements.

**Specific Analysis Methods:**

- **Growth Metrics**: CAGR, month-over-month growth, user acquisition trends
- **Engagement Metrics**: Editors-per-pageview ratio, session consistency (autocorrelation)
- **Stability Metrics**: Volatility, coefficient of variation, outlier frequency
- **Competitive Position**: Mobile app share of total mobile traffic (app vs web)
- **Trend Health**: Changepoint detection for negative shifts, growth rate evolution

#### Acceptance Criteria

1. WHEN assessing mobile app growth, THE Time_Series_Engine SHALL calculate: (a) CAGR from 2015-2025 with confidence intervals, (b) rolling 12-month growth rates to identify acceleration/deceleration, (c) comparison to mobile-web CAGR to assess competitive position

2. WHEN evaluating engagement quality, THE Multi_Dimensional_Analyzer SHALL calculate editors-per-million-pageviews ratio for mobile app and compare it to mobile-web and desktop using ANOVA to determine if app users show different engagement depth

3. WHEN measuring stability, THE Analysis_System SHALL compute: (a) coefficient of variation for monthly pageviews, (b) volatility (std of month-over-month changes), (c) frequency of outliers (values > 3 standard deviations from mean), and compare these metrics to mobile-web benchmarks

4. WHEN analyzing competitive position, THE Analysis_System SHALL calculate mobile app's share of total mobile traffic (app / (app + mobile-web)) over time and test whether this share is increasing, stable, or decreasing using trend analysis

5. WHEN detecting health issues, THE Time_Series_Engine SHALL apply changepoint detection algorithms to identify negative structural breaks in mobile app traffic and test their statistical significance

6. WHEN assessing trend health, THE Statistical_Validator SHALL test whether mobile app growth rate is significantly different from zero in recent periods (last 12 months, last 24 months) using t-tests

7. WHEN evaluating seasonality, THE Time_Series_Engine SHALL perform seasonal decomposition and test whether mobile app shows different seasonal patterns than mobile-web (e.g., stronger weekend effects, different holiday patterns)

8. WHEN cross-validating app health, THE Evidence_Framework SHALL correlate mobile app pageview trends with mobile app editor trends to ensure traffic changes reflect genuine engagement changes

9. WHEN benchmarking performance, THE Analysis_System SHALL compare mobile app metrics to industry benchmarks for mobile app engagement and retention where available

10. WHEN reporting mobile app health, THE Analysis_System SHALL provide: (a) growth metrics with trend classification (growing/stable/declining), (b) engagement quality comparison to other platforms, (c) stability assessment with benchmarks, (d) competitive position within mobile ecosystem, (e) identified health issues with changepoint analysis, (f) seasonal pattern analysis, (g) overall health score with recommendations

### Requirement 9: Weekday vs Weekend Usage Analysis (Research Question 8)

**User Story:** As a product analyst, I want to understand weekday vs weekend usage patterns with statistical validation, so that I can determine whether Wikipedia serves primarily utility (weekday) or leisure (weekend) needs and inform content strategy.

**Specific Analysis Methods:**

- **Data Granularity**: Daily pageviews to enable day-of-week aggregation
- **Statistical Test**: ANOVA for day-of-week effects, post-hoc tests for pairwise comparisons
- **Effect Size**: Cohen's d for weekday vs weekend difference magnitude
- **Engagement Comparison**: Editors-per-pageview ratio on weekdays vs weekends
- **Content Analysis**: Top articles on weekdays vs weekends to infer usage intent

#### Acceptance Criteria

1. WHEN analyzing day-of-week patterns, THE Data_Acquisition_Module SHALL fetch daily pageview data using `/metrics/pageviews/aggregate/en.wikipedia/all-access/user/daily/{start}/{end}` for the full analysis period

2. WHEN aggregating by day-of-week, THE Analysis_System SHALL categorize each date as Monday-Sunday and calculate mean pageviews for each day of week with 95% confidence intervals

3. WHEN testing for day-of-week effects, THE Statistical_Validator SHALL perform one-way ANOVA with day-of-week as factor to test whether mean pageviews differ significantly across days (H0: all days equal)

4. WHEN quantifying weekday vs weekend differences, THE Statistical_Validator SHALL: (a) aggregate Monday-Friday as "weekday" and Saturday-Sunday as "weekend", (b) perform t-test to compare means, (c) calculate Cohen's d effect size, (d) compute percentage difference with confidence intervals

5. WHEN analyzing engagement patterns, THE Multi_Dimensional_Analyzer SHALL calculate editors-per-million-pageviews ratio separately for weekdays and weekends and test whether engagement depth differs using t-tests (hypothesis: higher engagement on weekdays suggests utility usage)

6. WHEN examining content patterns, THE Analysis_System SHALL use Top Pages API to identify top 100 articles on weekdays vs weekends and categorize them as: reference/educational (utility), entertainment/current events (leisure), or mixed

7. WHEN testing temporal stability, THE Statistical_Validator SHALL test whether weekday vs weekend patterns have changed over time by comparing effect sizes across different years using meta-regression

8. WHEN controlling for seasonality, THE Time_Series_Engine SHALL perform seasonal decomposition and analyze day-of-week effects on the deseasonalized series to ensure patterns are not artifacts of seasonal variation

9. WHEN reporting weekday vs weekend findings, THE Analysis_System SHALL provide: (a) day-of-week mean pageviews with ANOVA results, (b) weekday vs weekend comparison with t-test, effect size, and percentage difference, (c) engagement depth comparison, (d) content category analysis, (e) temporal stability assessment, (f) interpretation of usage type (utility vs leisure) with supporting evidence

### Requirement 10: Platform Growth Driver Identification (Research Question 9)

**User Story:** As a strategic planner, I want to identify which platforms (desktop, mobile-web, mobile-app) are driving overall Wikipedia growth or decline, so that I can allocate resources to high-impact platforms and address declining platforms.

**Specific Analysis Methods:**

- **Decomposition**: Break down total traffic change into platform-specific contributions
- **Growth Attribution**: Calculate each platform's contribution to total growth using shift-share analysis
- **Trend Comparison**: Compare platform-specific CAGRs to overall CAGR
- **Marginal Analysis**: Assess how changes in each platform affect total traffic
- **Scenario Modeling**: Project total traffic under different platform growth scenarios

#### Acceptance Criteria

1. WHEN decomposing total traffic changes, THE Analysis_System SHALL calculate year-over-year change in total pageviews and attribute this change to each platform using: Δtotal = Δdesktop + Δmobile-web + Δmobile-app

2. WHEN performing shift-share analysis, THE Analysis_System SHALL decompose each platform's contribution into: (a) growth effect (platform growing/shrinking), (b) share effect (platform gaining/losing market share), (c) interaction effect

3. WHEN comparing growth rates, THE Statistical_Validator SHALL calculate CAGR for each platform and for total traffic, then test whether each platform's CAGR differs significantly from the total CAGR using bootstrap confidence intervals

4. WHEN identifying growth drivers, THE Analysis_System SHALL rank platforms by their absolute contribution to total traffic change (in pageviews) and their relative contribution (percentage of total change)

5. WHEN analyzing marginal effects, THE Statistical_Validator SHALL calculate the correlation between each platform's month-over-month change and total traffic's month-over-month change to identify which platform changes most strongly predict total traffic changes

6. WHEN testing platform independence, THE Statistical_Validator SHALL test whether platform growth rates are correlated (suggesting common drivers) or independent (suggesting platform-specific drivers) using correlation tests

7. WHEN projecting future scenarios, THE Time_Series_Engine SHALL model total traffic under scenarios: (a) all platforms continue current trends, (b) declining platforms stabilize, (c) growing platforms accelerate, and calculate resulting total traffic with confidence intervals

8. WHEN identifying inflection points, THE Time_Series_Engine SHALL detect when each platform transitioned from growth to decline (or vice versa) using changepoint detection and assess whether these transitions align across platforms or are platform-specific

9. WHEN reporting growth drivers, THE Analysis_System SHALL provide: (a) shift-share decomposition table showing each platform's contribution, (b) growth rate comparison with significance tests, (c) platform ranking by contribution to total change, (d) marginal effect analysis, (e) correlation matrix of platform growth rates, (f) scenario projections, (g) inflection point timeline, (h) strategic recommendations for platform investment

### Requirement 11: Long-term User Reliance Trend Analysis (Research Question 10)

**User Story:** As a strategic analyst, I want to understand how user reliance on Wikipedia has evolved over the long term, so that I can assess whether Wikipedia remains essential to users or is being displaced by alternative information sources.

**Specific Analysis Methods:**

- **Reliance Proxies**: Pageviews per internet user (using global internet penetration data), engagement depth (editors/pageviews), content freshness (edits/pageviews)
- **Trend Analysis**: Long-term trend extraction using Hodrick-Prescott filter or polynomial regression
- **Structural Break Detection**: Identify periods where reliance fundamentally shifted
- **Cross-Validation**: Correlate with external data (Google Trends for "Wikipedia", direct traffic vs search referrals if available)

#### Acceptance Criteria

1. WHEN measuring absolute reliance, THE Analysis_System SHALL calculate total monthly pageviews and test for long-term trends using Mann-Kendall trend test (non-parametric, robust to outliers)

2. WHEN normalizing for internet growth, THE Analysis_System SHALL acquire global internet user statistics from ITU or World Bank and calculate pageviews-per-million-internet-users as a reliance metric, then test whether this normalized metric shows increasing, stable, or decreasing trends

3. WHEN assessing engagement depth, THE Multi_Dimensional_Analyzer SHALL calculate the ratio of active editors to pageviews over time and test whether this ratio is increasing (deeper engagement) or decreasing (more passive consumption) using trend analysis

4. WHEN measuring content vitality, THE Multi_Dimensional_Analyzer SHALL calculate the ratio of edits to pageviews over time as a proxy for content freshness and user investment, then test for significant trends

5. WHEN detecting reliance shifts, THE Time_Series_Engine SHALL apply changepoint detection to the normalized reliance metrics and identify periods where user reliance fundamentally changed, testing each changepoint for statistical significance

6. WHEN cross-validating with external data, THE Evidence_Framework SHALL correlate Wikipedia pageview trends with Google Trends data for "Wikipedia" searches to assess whether interest in Wikipedia as a brand aligns with usage trends

7. WHEN analyzing platform-specific reliance, THE Analysis_System SHALL repeat reliance analysis separately for desktop, mobile-web, and mobile-app to test whether reliance patterns differ by platform (e.g., mobile users may show different reliance patterns)

8. WHEN comparing to competitors, THE Analysis_System SHALL benchmark Wikipedia's growth against other reference platforms (if data available) to assess relative market position

9. WHEN reporting reliance trends, THE Analysis_System SHALL provide: (a) absolute pageview trends with Mann-Kendall test results, (b) normalized pageviews-per-internet-user trends, (c) engagement depth evolution, (d) content vitality trends, (e) structural break analysis with changepoint dates, (f) cross-validation with Google Trends, (g) platform-specific reliance patterns, (h) interpretation of whether user reliance is increasing, stable, or decreasing with supporting evidence

### Requirement 12: Global Disruption Response Analysis (Research Question 11)

**User Story:** As a content strategist, I want to measure how Wikipedia usage responds to major global disruptions and news cycles with quantified sensitivity, so that I can understand Wikipedia's role during major events and optimize content strategy for high-impact periods.

**Specific Analysis Methods:**

- **Event Study Methodology**: Construct baseline forecasts, measure deviations during events
- **Event Categories**: Political (elections, conflicts), natural disasters, celebrity deaths, scientific breakthroughs, pandemics
- **Impact Metrics**: Cumulative Abnormal Return (CAR), peak impact magnitude, duration, decay half-life
- **Statistical Tests**: Test whether observed values exceed 95% prediction intervals
- **Cross-Event Comparison**: ANOVA to test if response magnitudes differ by event category

#### Acceptance Criteria

1. WHEN identifying major events, THE Analysis_System SHALL compile a list of significant global disruptions from 2015-2025 including: (a) political events (US elections 2016/2020/2024, Brexit 2016, major conflicts), (b) natural disasters (major earthquakes, hurricanes, wildfires), (c) celebrity deaths (high-profile figures), (d) scientific events (Nobel prizes, space missions), (e) pandemic (COVID-19 2020-2021), categorized by type

2. WHEN analyzing an event, THE Causal_Analyzer SHALL implement event study methodology: (a) define event window (typically ±30 days around event date), (b) define estimation window (90 days pre-event for baseline), (c) fit ARIMA or Prophet model on estimation window, (d) forecast expected traffic during event window with 95% prediction intervals

3. WHEN measuring event impact, THE Statistical_Validator SHALL calculate: (a) daily abnormal returns (observed - predicted), (b) cumulative abnormal return (CAR = sum of daily abnormal returns), (c) peak impact (maximum daily abnormal return), (d) impact duration (number of days with significant abnormal returns)

4. WHEN testing significance, THE Statistical_Validator SHALL determine whether observed values exceed the 95% prediction interval and calculate z-scores for the magnitude of deviation

5. WHEN measuring persistence, THE Time_Series_Engine SHALL fit exponential decay model to post-event abnormal returns and estimate half-life (time for impact to decay to 50% of peak)

6. WHEN comparing event categories, THE Statistical_Validator SHALL perform ANOVA to test whether CAR, peak impact, or duration differ significantly across event categories (political, natural disaster, celebrity, scientific, pandemic)

7. WHEN analyzing content response, THE Analysis_System SHALL use Top Pages API to identify which articles saw largest traffic increases during each event and categorize them to understand what content users seek during disruptions

8. WHEN cross-validating with engagement, THE Multi_Dimensional_Analyzer SHALL check whether event-driven pageview spikes align with editor activity spikes (suggesting active content creation) or remain isolated to pageviews (passive consumption)

9. WHEN testing platform differences, THE Analysis_System SHALL repeat event analysis separately for desktop, mobile-web, and mobile-app to test whether different platforms show different event sensitivities

10. WHEN correlating with news intensity, THE Analysis_System SHALL attempt to correlate event impact magnitude with external news cycle intensity metrics (e.g., Google Trends for event-related terms, news article counts) to test whether Wikipedia response scales with event prominence

11. WHEN reporting event response findings, THE Analysis_System SHALL provide: (a) event catalog with dates and categories, (b) observed vs predicted plots for major events with prediction intervals, (c) CAR, peak impact, and duration for each event, (d) z-scores and significance tests, (e) decay half-life analysis, (f) cross-category comparison with ANOVA results, (g) content analysis showing what users seek during events, (h) engagement quality assessment, (i) platform-specific response patterns, (j) correlation with news intensity, (k) interpretation of Wikipedia's role during major events

### Requirement 13: Statistical Validation Framework

**User Story:** As a data scientist, I want to validate all trend changes and patterns with statistical significance tests, so that I can distinguish real effects from random variation across all research questions.

#### Acceptance Criteria

1. WHEN analyzing a trend change, THE Statistical_Validator SHALL perform hypothesis testing to determine if the change is statistically significant at α = 0.05

2. WHEN conducting hypothesis tests, THE Statistical_Validator SHALL calculate p-values and compare them against the defined Significance_Level

3. WHEN evaluating impacts (campaigns, events, structural breaks), THE Statistical_Validator SHALL compute effect sizes with 95% Confidence_Intervals using appropriate methods (Cohen's d for mean differences, percentage change for proportions)

4. WHEN generating forecasts, THE Statistical_Validator SHALL provide prediction intervals at 50%, 80%, and 95% confidence levels alongside point estimates

5. WHEN comparing groups (platforms, time periods, event categories), THE Statistical_Validator SHALL perform appropriate statistical tests: t-tests for two groups, ANOVA for multiple groups, or non-parametric equivalents (Mann-Whitney, Kruskal-Wallis) if assumptions violated

6. WHEN detecting structural shifts, THE Statistical_Validator SHALL apply changepoint detection algorithms (PELT, Binary Segmentation, Bayesian) with significance thresholds and require consensus from at least two methods

7. WHEN reporting results, THE Statistical_Validator SHALL include test statistics, p-values, confidence intervals, and effect sizes for all findings, with plain-language interpretation of statistical significance and practical significance

### Requirement 14: Cross-Validation and Robustness

**User Story:** As a research analyst, I want to validate findings across multiple data sources and methodologies, so that I can ensure conclusions are robust and not artifacts of a single approach.

#### Acceptance Criteria

1. WHEN validating a finding, THE Evidence_Framework SHALL confirm the pattern across multiple data sources: pageviews, active editors, and edit volume

2. WHEN detecting platform-specific patterns, THE Evidence_Framework SHALL verify consistency across desktop, mobile-web, and mobile-app data

3. WHEN testing robustness, THE Evidence_Framework SHALL perform sensitivity analysis by varying model parameters (e.g., changepoint penalty, forecast horizon, significance level) and confirm findings remain stable

4. WHEN handling outliers, THE Evidence_Framework SHALL test findings with and without outlier removal (using z-score > 3 threshold) to assess impact on conclusions

5. WHEN adjusting for seasonality, THE Evidence_Framework SHALL validate seasonal decomposition using multiple methods (STL, X-13-ARIMA-SEATS) and confirm consistency

6. WHEN comparing methods, THE Evidence_Framework SHALL apply multiple analytical approaches to the same question (e.g., multiple changepoint algorithms, multiple forecasting models) and report consensus findings

7. WHEN reporting validation results, THE Evidence_Framework SHALL provide consistency scores indicating what proportion of validation checks support each finding

### Requirement 15: Reproducible Analysis Pipeline

**User Story:** As a data engineer, I want a reproducible analysis pipeline with version control and complete documentation, so that all findings can be independently verified, audited, and updated with new data.

#### Acceptance Criteria

1. WHEN executing analysis, THE Analysis_System SHALL log all data sources with: (a) API endpoints (e.g., `https://wikimedia.org/api/rest_v1/metrics/pageviews/aggregate/en.wikipedia/desktop/user/monthly/2019010100/2024123100`), (b) query parameters (project, access, agent, granularity, date range), (c) acquisition timestamps, (d) data version identifiers or checksums

2. WHEN performing statistical tests, THE Analysis_System SHALL record: (a) exact test names and implementations (e.g., "scipy.stats.ttest_ind version 1.11.0"), (b) test assumptions and validation results, (c) significance levels used, (d) all parameter values

3. WHEN generating results, THE Analysis_System SHALL save intermediate outputs (cleaned data, model objects, test results) and final results with SHA-256 checksums for integrity verification

4. WHEN documenting analysis, THE Analysis_System SHALL produce reports including: (a) code commit hashes, (b) dependency versions (Python, pandas, scipy, statsmodels, prophet versions), (c) random seeds for reproducibility, (d) parameter specifications, (e) execution environment details (OS, Python version)

5. WHEN updating data, THE Analysis_System SHALL support re-running the entire pipeline with new data inputs while preserving historical analysis versions in separate directories with timestamps

6. WHEN validating reproducibility, THE Analysis_System SHALL produce identical results (within numerical precision tolerance of 1e-10) when run with the same inputs, parameters, and random seeds

7. WHEN versioning analysis, THE Analysis_System SHALL track changes to methods, data sources, and parameters using git tags and maintain a CHANGELOG.md with rationale for changes

### Requirement 16: Evidence-Backed Visualization and Reporting

**User Story:** As a stakeholder, I want visualizations that display statistical evidence alongside findings with clear interpretation guidance, so that I can assess the strength of conclusions at a glance and understand what the evidence means.

#### Acceptance Criteria

1. WHEN displaying trends, THE Analysis_System SHALL include 95% confidence bands or prediction intervals on all time series plots with shaded regions and legend indicating confidence level

2. WHEN showing campaign or event effects, THE Analysis_System SHALL visualize: (a) observed data as solid line, (b) counterfactual baseline as dashed line, (c) confidence intervals as shaded regions, (d) annotations with effect size and p-value

3. WHEN presenting statistical tests, THE Analysis_System SHALL annotate visualizations with: (a) p-values formatted as "p < 0.001", "p = 0.023", or "p = 0.156 (ns)", (b) effect sizes with interpretation (small/medium/large based on Cohen's conventions), (c) significance indicators (*, **, ***)

4. WHEN comparing groups, THE Analysis_System SHALL display error bars or confidence intervals for all comparative metrics and indicate significant differences with connecting lines and p-values

5. WHEN showing forecasts, THE Analysis_System SHALL plot: (a) historical data in one color, (b) point forecasts in another color, (c) uncertainty fans with 50%, 80%, 95% intervals in progressively lighter shades, (d) legend explaining confidence levels

6. WHEN reporting findings, THE Analysis_System SHALL generate summary tables with columns: (a) metric name, (b) value, (c) confidence interval, (d) test statistic, (e) p-value, (f) effect size, (g) interpretation

7. WHEN creating dashboards, THE Analysis_System SHALL provide interactive elements: (a) hover tooltips showing confidence intervals and sample sizes, (b) click-through to detailed test results, (c) date range filters, (d) platform toggles, (e) methodology tooltips explaining each statistical measure

8. WHEN presenting research question answers, THE Analysis_System SHALL structure reports as: (a) research question statement, (b) data sources and methods used, (c) key findings with statistical evidence, (d) visualizations with evidence overlays, (e) robustness checks, (f) limitations and caveats, (g) actionable recommendations

### Requirement 17: API Rate Limiting and Error Handling

**User Story:** As a system operator, I want robust API rate limiting and error handling, so that the system can reliably acquire data from Wikimedia APIs without failures or service disruptions.

#### Acceptance Criteria

1. WHEN making API requests, THE Data_Acquisition_Module SHALL implement rate limiting to respect Wikimedia API limits (200 requests per second per IP, with user-agent identification)

2. WHEN an API request fails, THE Data_Acquisition_Module SHALL implement exponential backoff retry logic with: (a) maximum 5 retries, (b) backoff factor of 2 (wait 1s, 2s, 4s, 8s, 16s), (c) jitter to avoid thundering herd

3. WHEN API responses are received, THE Data_Acquisition_Module SHALL validate response status codes and raise appropriate errors: (a) 200 OK → process data, (b) 404 Not Found → log and skip, (c) 429 Too Many Requests → backoff and retry, (d) 500 Server Error → retry with backoff

4. WHEN data is incomplete, THE Data_Acquisition_Module SHALL log missing date ranges and attempt to fill gaps with additional API calls, up to 3 attempts per gap

5. WHEN all retries fail, THE Data_Acquisition_Module SHALL raise DataAcquisitionError with detailed context (endpoint, parameters, error messages, timestamps) and fall back to cached data if available and within staleness threshold (24 hours)

6. WHEN caching data, THE Data_Acquisition_Module SHALL store API responses with metadata (acquisition timestamp, endpoint, parameters) and use cached data only when fresh data acquisition fails

7. WHEN logging API interactions, THE Data_Acquisition_Module SHALL record: (a) all requests with timestamps, (b) response status codes, (c) response times, (d) retry attempts, (e) errors encountered, for monitoring and debugging

## Additional Analysis Opportunities

This section suggests valuable extensions and subtopics that could enhance the analysis:

### Geographic Analysis

**Opportunity**: Analyze Wikipedia usage patterns across different geographic regions to understand global vs regional trends.

**Data Source**: Wikimedia API supports country-level pageview data via `/metrics/pageviews/top-by-country/{project}/{access}/{year}/{month}`

**Analysis Ideas**:
- Compare AI search impact across regions (hypothesis: stronger impact in English-speaking countries)
- Identify regions driving growth vs decline
- Analyze event response by region (local events vs global events)
- Test whether mobile dependency varies by region (hypothesis: higher in developing countries)

### Language Edition Comparison

**Opportunity**: Compare English Wikipedia to other major language editions to understand language-specific patterns.

**Data Source**: Replace `en.wikipedia` with other project codes (e.g., `es.wikipedia`, `de.wikipedia`, `fr.wikipedia`)

**Analysis Ideas**:
- Test whether AI search impact is English-specific or global
- Compare mobile adoption rates across languages
- Identify language-specific growth drivers
- Analyze whether engagement patterns differ by language/culture

### Article-Level Analysis

**Opportunity**: Analyze top articles to understand what content drives traffic and how it changes over time.

**Data Source**: Top Pages API `/metrics/pageviews/top/{project}/{access}/{year}/{month}/{day}` returns top 1000 articles

**Analysis Ideas**:
- Categorize top articles (news, reference, entertainment, education) and track category shifts
- Identify "evergreen" articles with stable traffic vs "spike" articles driven by events
- Analyze whether top articles show different platform preferences
- Test whether article diversity (concentration of traffic) has changed over time

### Referral Source Analysis

**Opportunity**: Understand how users arrive at Wikipedia (search engines, direct, social media, other referrals).

**Data Source**: May require Wikimedia Analytics datasets or server logs (not available via public API)

**Analysis Ideas**:
- Test whether search referrals have declined with AI search rise
- Analyze whether direct traffic (bookmarks, habitual users) shows more stability
- Identify which referral sources drive engaged users vs passive readers
- Track social media referral trends over time

### Content Freshness Analysis

**Opportunity**: Analyze the relationship between content updates and pageviews to understand content vitality.

**Data Source**: Edit volume data from Wikimedia Analytics, correlated with pageview data

**Analysis Ideas**:
- Test whether articles with recent edits receive more pageviews
- Analyze whether edit frequency predicts traffic stability
- Identify whether declining traffic correlates with declining edit activity
- Test whether campaigns drive both pageviews and edits (true engagement)

### Seasonal Deep Dive

**Opportunity**: Detailed analysis of seasonal patterns beyond basic weekday/weekend.

**Analysis Ideas**:
- Identify specific holidays with largest Wikipedia usage spikes
- Analyze back-to-school effects (September) and summer effects (June-August)
- Test whether seasonal patterns have changed over time (e.g., summer dip shrinking with mobile adoption)
- Compare seasonal patterns across platforms (hypothesis: mobile shows weaker seasonality)

### User Cohort Analysis

**Opportunity**: Analyze whether Wikipedia is retaining existing users or acquiring new users.

**Data Source**: Requires user-level data (may not be publicly available due to privacy)

**Analysis Ideas**:
- Estimate user retention using statistical models on aggregate data
- Test whether traffic decline is due to user churn or reduced usage per user
- Analyze whether new user acquisition has slowed
- Identify platform-specific retention patterns

### Competitive Benchmarking

**Opportunity**: Compare Wikipedia's trends to other reference and information platforms.

**Data Source**: SimilarWeb, Alexa (historical), Google Trends, or other web analytics platforms

**Analysis Ideas**:
- Compare Wikipedia's growth to Stack Overflow, Quora, Reddit
- Test whether Wikipedia's decline is unique or part of broader "web search decline"
- Benchmark mobile adoption rates against industry averages
- Analyze whether Wikipedia is losing share to video platforms (YouTube) for educational content

### Search Query Analysis

**Opportunity**: Analyze what users search for on Wikipedia to understand information needs.

**Data Source**: Wikimedia Search API or search logs (if available)

**Analysis Ideas**:
- Identify trending search topics over time
- Analyze whether search queries have become more specific or more general
- Test whether search behavior differs by platform
- Correlate search trends with pageview trends

### Performance and Technical Metrics

**Opportunity**: Analyze whether technical performance affects usage patterns.

**Data Source**: Wikimedia performance metrics, page load times (if available)

**Analysis Ideas**:
- Test whether page load time correlates with bounce rate or session duration
- Analyze whether mobile app performance differs from mobile web
- Identify whether technical issues coincide with traffic drops
- Test whether performance improvements drive usage increases

### Accessibility and Internationalization

**Opportunity**: Analyze usage patterns related to accessibility features and international users.

**Analysis Ideas**:
- Test whether mobile app accessibility features drive engagement
- Analyze usage patterns in regions with limited internet connectivity
- Identify whether Wikipedia's role differs in information-scarce vs information-rich regions
- Test whether multilingual users show different engagement patterns

### Machine Learning and Prediction

**Opportunity**: Apply advanced ML techniques for prediction and pattern discovery.

**Analysis Ideas**:
- Use LSTM or Transformer models for long-term forecasting
- Apply anomaly detection algorithms to identify unusual patterns automatically
- Use clustering to identify distinct user behavior segments
- Apply causal discovery algorithms to identify causal relationships in data

## Wikimedia API Reference

This section provides detailed reference for all Wikimedia APIs used in the analysis.

### Pageviews API

**Base URL**: `https://wikimedia.org/api/rest_v1/`

**Documentation**: https://wikimedia.org/api/rest_v1/#/Pageviews_data

#### Aggregate Pageviews Endpoint

**Endpoint**: `/metrics/pageviews/aggregate/{project}/{access}/{agent}/{granularity}/{start}/{end}`

**Parameters**:
- `project`: Wikipedia project (e.g., `en.wikipedia`, `en.wikipedia.org`, `all-projects`)
- `access`: Access method - `desktop`, `mobile-web`, `mobile-app`, or `all-access`
- `agent`: Agent type - `user` (human), `spider` (bot), or `all-agents`
- `granularity`: Time granularity - `hourly`, `daily`, or `monthly`
- `start`: Start date in format `YYYYMMDDHH` (hour optional for daily/monthly)
- `end`: End date in format `YYYYMMDDHH`

**Example Request**:
```
GET https://wikimedia.org/api/rest_v1/metrics/pageviews/aggregate/en.wikipedia/desktop/user/monthly/2019010100/2024123100
```

**Response Format**:
```json
{
  "items": [
    {
      "project": "en.wikipedia",
      "access": "desktop",
      "agent": "user",
      "granularity": "monthly",
      "timestamp": "2019010100",
      "views": 8234567890
    },
    ...
  ]
}
```

**Rate Limits**: 200 requests per second per IP

**Best Practices**:
- Use monthly granularity for long-term trend analysis (reduces API calls)
- Use daily granularity for event analysis and day-of-week patterns
- Always set agent=user to exclude bot traffic
- Make separate calls for each access method to enable platform comparison
- Include User-Agent header identifying your application

#### Top Pages Endpoint

**Endpoint**: `/metrics/pageviews/top/{project}/{access}/{year}/{month}/{day}`

**Parameters**:
- `project`: Wikipedia project (e.g., `en.wikipedia`)
- `access`: Access method - `desktop`, `mobile-web`, `mobile-app`, or `all-access`
- `year`: Year (YYYY)
- `month`: Month (MM)
- `day`: Day (DD)

**Example Request**:
```
GET https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/2024/01/15
```

**Response Format**:
```json
{
  "items": [
    {
      "project": "en.wikipedia",
      "access": "all-access",
      "year": "2024",
      "month": "01",
      "day": "15",
      "articles": [
        {
          "article": "Main_Page",
          "views": 12345678,
          "rank": 1
        },
        ...
      ]
    }
  ]
}
```

**Returns**: Top 1000 articles by pageviews for the specified day

**Use Cases**:
- Content analysis (what users read during events)
- Weekday vs weekend content comparison
- Trending topic identification

#### Per-Article Pageviews Endpoint

**Endpoint**: `/metrics/pageviews/per-article/{project}/{access}/{agent}/{article}/{granularity}/{start}/{end}`

**Parameters**:
- `project`: Wikipedia project
- `access`: Access method
- `agent`: Agent type
- `article`: Article title (URL-encoded)
- `granularity`: Time granularity
- `start`: Start date
- `end`: End date

**Example Request**:
```
GET https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/Albert_Einstein/daily/20240101/20240131
```

**Use Cases**:
- Article-level trend analysis
- Event impact on specific articles
- Content performance tracking

### Wikimedia Analytics Datasets

**Access**: https://dumps.wikimedia.org/other/analytics/

**Available Datasets**:

#### Editor Metrics
- **Path**: `/other/analytics/editor_metrics/`
- **Format**: TSV files with monthly editor counts
- **Columns**: project, month, active_editors, new_editors, returning_editors
- **Granularity**: Monthly
- **Use**: Cross-validation of pageview trends with editor engagement

#### Edit Counts
- **Path**: `/other/analytics/edit_counts/`
- **Format**: TSV files with edit volume
- **Columns**: project, month, total_edits, registered_edits, anonymous_edits
- **Granularity**: Monthly
- **Use**: Content vitality analysis, engagement depth measurement

#### Pageview Complete Dumps
- **Path**: `/other/pageviews/`
- **Format**: Gzipped text files with hourly pageviews
- **Granularity**: Hourly
- **Use**: Detailed analysis requiring hourly data, custom aggregations

### XTools API (Alternative for Editor Data)

**Base URL**: `https://xtools.wmflabs.org/api/`

**Documentation**: https://www.mediawiki.org/wiki/XTools/API

**Endpoints**:
- `/project/pages_created/{project}`: Pages created over time
- `/project/automated_edits/{project}`: Automated edit statistics
- `/user/simple_editcount/{project}/{username}`: User edit counts

**Note**: XTools provides user-level and article-level metrics but may have rate limits

### Data Acquisition Strategy

**Recommended Approach**:

1. **Initial Data Collection**:
   - Use Pageviews API for monthly aggregated data (2015-2025)
   - Make separate calls for desktop, mobile-web, mobile-app
   - Download editor metrics from Wikimedia Analytics datasets
   - Download edit volume data from Wikimedia Analytics datasets

2. **Incremental Updates**:
   - Schedule monthly updates to fetch latest data
   - Use daily granularity for recent months to enable timely event analysis
   - Aggregate daily to monthly for consistency with historical data

3. **Caching Strategy**:
   - Cache all API responses with metadata (timestamp, parameters)
   - Store processed data in efficient format (Parquet, HDF5)
   - Implement cache invalidation based on data staleness (24 hours)

4. **Error Handling**:
   - Implement exponential backoff for rate limit errors
   - Log all failed requests for manual review
   - Fall back to cached data if API unavailable
   - Validate response completeness before processing

5. **Rate Limit Management**:
   - Implement request throttling (max 100 requests/second to be safe)
   - Add delays between requests (10ms minimum)
   - Use connection pooling for efficiency
   - Monitor rate limit headers in responses

### API Response Validation

**Required Checks**:

1. **Status Code Validation**:
   - 200 OK: Process response
   - 404 Not Found: Log and skip (data may not exist for date range)
   - 429 Too Many Requests: Backoff and retry
   - 500 Server Error: Retry with exponential backoff

2. **Data Completeness**:
   - Verify all requested dates present in response
   - Check for null or zero values (may indicate data issues)
   - Validate timestamp format and ordering
   - Ensure views field is numeric and non-negative

3. **Schema Validation**:
   - Verify expected fields present (project, access, agent, timestamp, views)
   - Check data types match expectations
   - Validate enum values (access, agent, granularity)

4. **Logical Validation**:
   - Check views are within reasonable ranges (e.g., < 100 billion per month)
   - Verify timestamps are within requested range
   - Ensure no duplicate timestamps
   - Validate that mobile-web + mobile-app ≈ all-mobile (within tolerance)

## Requirements Traceability Matrix

This matrix maps each research question to the requirements that address it:

| Research Question | Primary Requirements | Supporting Requirements |
|------------------|---------------------|------------------------|
| Q1: AI-Assisted Search Impact | Req 2 | Req 1, 13, 14 |
| Q2: Future Traffic Projections | Req 3 | Req 1, 13, 14 |
| Q3: Mobile Dependency Risk | Req 4 | Req 1, 13, 14 |
| Q4: Usage Pattern Changes | Req 5 | Req 1, 13, 14 |
| Q5: Mobile App vs Web Stability | Req 6 | Req 1, 13, 14 |
| Q6: Campaign Effectiveness | Req 7 | Req 1, 13, 14 |
| Q7: Mobile App Product Health | Req 8 | Req 1, 13, 14 |
| Q8: Weekday vs Weekend Usage | Req 9 | Req 1, 13, 14 |
| Q9: Platform Growth Drivers | Req 10 | Req 1, 13, 14 |
| Q10: Long-term Reliance Trends | Req 11 | Req 1, 13, 14 |
| Q11: Global Disruption Response | Req 12 | Req 1, 13, 14 |

**Cross-Cutting Requirements**:
- **Req 1**: Data acquisition foundation for all analyses
- **Req 13**: Statistical validation applied to all findings
- **Req 14**: Cross-validation ensures robustness of all conclusions
- **Req 15**: Reproducibility enables verification of all analyses
- **Req 16**: Visualization makes all findings accessible
- **Req 17**: Error handling ensures reliable data acquisition

## Implementation Priority

**Phase 1: Foundation (Weeks 1-2)**
- Req 1: Data Acquisition Module
- Req 17: API Error Handling
- Req 15: Reproducible Pipeline Setup

**Phase 2: Core Statistical Framework (Weeks 3-4)**
- Req 13: Statistical Validation Engine
- Req 14: Cross-Validation Framework
- Req 16: Basic Visualization

**Phase 3: Research Questions - High Priority (Weeks 5-8)**
- Req 2: AI Search Impact (highest stakeholder interest)
- Req 3: Future Projections (critical for planning)
- Req 4: Mobile Dependency (strategic risk)
- Req 10: Platform Growth Drivers (resource allocation)

**Phase 4: Research Questions - Medium Priority (Weeks 9-12)**
- Req 5: Usage Pattern Evolution
- Req 8: Mobile App Health
- Req 11: Long-term Reliance
- Req 12: Event Response

**Phase 5: Research Questions - Lower Priority (Weeks 13-14)**
- Req 6: Mobile App vs Web Stability
- Req 7: Campaign Effectiveness
- Req 9: Weekday vs Weekend

**Phase 6: Enhancement (Weeks 15-16)**
- Advanced visualizations
- Interactive dashboards
- Additional analysis opportunities
- Documentation and knowledge transfer

## Success Criteria

The Wikipedia Product Health Analysis system will be considered successful when:

1. **Data Completeness**: All pageview, editor, and edit data from 2015-2025 acquired with < 1% missing values

2. **Statistical Rigor**: Every finding includes p-values, confidence intervals, and effect sizes; no claims without statistical evidence

3. **Research Questions Answered**: All 11 research questions answered with concrete, evidence-backed conclusions

4. **Reproducibility**: Analysis can be re-run with identical results (within 1e-10 tolerance) given same inputs

5. **Cross-Validation**: All major findings validated across multiple data sources with consistency scores > 0.8

6. **Stakeholder Confidence**: Findings presented with clear visualizations showing statistical evidence, enabling data-driven decisions

7. **Actionability**: Each research question produces specific, actionable recommendations supported by evidence

8. **Robustness**: Findings remain stable under sensitivity analysis (parameter variations, outlier removal)

9. **Timeliness**: Analysis pipeline can be updated with new data monthly within 1 hour execution time

10. **Accessibility**: Non-technical stakeholders can understand findings through clear visualizations and plain-language interpretations

## Glossary Additions

Additional terms specific to the expanded requirements:

- **CAGR**: Compound Annual Growth Rate - geometric mean growth rate over multiple periods
- **Shift-Share_Analysis**: Decomposition method attributing total change to component-specific contributions
- **Mann-Kendall_Test**: Non-parametric test for monotonic trends in time series
- **Levene_Test**: Test for equality of variances across groups
- **Autocorrelation**: Correlation of a time series with itself at different time lags
- **Half-Life**: Time required for an effect to decay to 50% of its initial magnitude
- **Cumulative_Abnormal_Return**: Sum of differences between observed and predicted values during an event window
- **Bootstrap**: Resampling method for estimating confidence intervals
- **Permutation_Test**: Non-parametric significance test based on random permutations
- **Hodrick-Prescott_Filter**: Method for separating trend from cyclical components in time series
- **Diebold-Mariano_Test**: Test for comparing forecast accuracy of different models
- **Chow_Test**: Test for structural breaks in regression models
- **PELT**: Pruned Exact Linear Time algorithm for changepoint detection
- **STL**: Seasonal and Trend decomposition using Loess
- **Prophet**: Facebook's time series forecasting library
- **ARIMA**: AutoRegressive Integrated Moving Average model for time series
- **HHI**: Herfindahl-Hirschman Index - measure of market concentration
- **Cohen_d**: Standardized measure of effect size for mean differences
- **MAPE**: Mean Absolute Percentage Error - forecast accuracy metric
- **RMSE**: Root Mean Squared Error - forecast accuracy metric
- **MAE**: Mean Absolute Error - forecast accuracy metric
- **MASE**: Mean Absolute Scaled Error - scale-independent forecast accuracy metric

## Notes on Data Privacy and Ethics

**Privacy Considerations**:
- All analysis uses aggregate data only (no individual user tracking)
- Pageview data is anonymized by Wikimedia before API exposure
- Editor counts are aggregated (no individual editor identification)
- Compliance with Wikimedia's data usage policies and terms of service

**Ethical Considerations**:
- Analysis aims to improve Wikipedia's service to users
- Findings will be shared with Wikimedia community
- No commercial exploitation of data
- Respect for Wikipedia's mission of free knowledge

**Data Usage Policy Compliance**:
- Follow Wikimedia API terms of service
- Include User-Agent header identifying analysis purpose
- Respect rate limits and implement backoff
- Cache data to minimize API load
- Attribute data source in all publications

## Conclusion

This requirements document specifies a comprehensive, rigorous approach to analyzing Wikipedia's product health. By combining specific data sources (Wikimedia APIs), concrete analytical methods (statistical tests, causal inference, time series analysis), and a multi-layered validation framework, the system will produce evidence-backed insights that distinguish real patterns from noise.

The 11 research questions address critical strategic concerns:
- Understanding AI search impact on traffic
- Planning for future capacity needs
- Managing platform dependency risks
- Tracking usage pattern evolution
- Optimizing platform investments
- Measuring campaign ROI
- Assessing mobile app health
- Understanding user behavior patterns
- Identifying growth drivers
- Evaluating long-term relevance
- Understanding Wikipedia's role during major events

Each requirement specifies not just what to analyze, but exactly how to analyze it: which API endpoints to call, which statistical tests to perform, which parameters to use, and which validation checks to apply. This level of specificity ensures the analysis can be implemented systematically and reproduced reliably.

The system's commitment to statistical rigor—requiring p-values, confidence intervals, effect sizes, and cross-validation for every finding—ensures that conclusions are defensible and actionable, addressing the core concern about "naive traffic analysis" by making statistical validation mandatory at every step.
