"""Insight Generator for Fortune 500 Knowledge Graph Analytics.

Implements:
- Underperformer identification by Innovation Score vs sector average (Requirement 9.1)
- Investment recommendations (Requirement 9.2, 9.4)
- Acquisition target identification (Requirement 9.3)
- ROI calculations (Requirement 10.1-10.5)
- Executive report generation (Requirement 11.1-11.5)
- Executive report export to PDF and HTML (Requirement 11.6)
- Best practice identification from high-performing sectors (Requirement 13.5)
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import os
import uuid

from .analytics_engine import MetricsRepository
from .data_models import (
    AcquisitionTarget,
    BestPractice,
    Company,
    DigitalMaturityRecord,
    EcosystemCentralityRecord,
    ExecutiveReport,
    InflectionPoint,
    InnovationScoreRecord,
    LeaderboardEntry,
    MetricRecord,
    MetricsSummary,
    Recommendation,
    ROIAnalysis,
    ROIMetrics,
    TrendsAnalysis,
)

logger = logging.getLogger(__name__)


def validate_recommendation(recommendation: "Recommendation") -> None:
    """Validate that a Recommendation object has all required fields populated.

    Checks:
    - supporting_metrics: non-empty dict
    - confidence_level: float in [0.0, 1.0]
    - expected_outcome: non-empty string

    Args:
        recommendation: The Recommendation object to validate.

    Raises:
        ValueError: If any required field is missing or invalid.

    Validates: Requirement 9.5
    """
    if not recommendation.supporting_metrics:
        raise ValueError(
            f"Recommendation '{recommendation.title}' must have non-empty supporting_metrics"
        )
    if not (0.0 <= recommendation.confidence_level <= 1.0):
        raise ValueError(
            f"Recommendation '{recommendation.title}' confidence_level must be in [0.0, 1.0], "
            f"got {recommendation.confidence_level}"
        )
    if not recommendation.expected_outcome or not recommendation.expected_outcome.strip():
        raise ValueError(
            f"Recommendation '{recommendation.title}' must have a non-empty expected_outcome"
        )


@dataclass
class UnderperformerResult:
    """A company identified as underperforming relative to its sector average.

    Attributes:
        company: The underperforming company.
        innovation_score: The company's normalized Innovation Score.
        sector_average: The sector's average normalized Innovation Score.
        gap: Difference between sector_average and the company's score
             (always positive; larger gap = further below average).
    """

    company: Company
    innovation_score: float
    sector_average: float
    gap: float


class InsightGenerator:
    """
    Generates business insights and recommendations from computed metrics.

    Requires a MetricsRepository populated with InnovationScoreRecord entries
    (produced by AnalyticsEngine.store_innovation_scores) and a list of Company
    objects so that sector membership can be resolved.
    """

    def __init__(
        self,
        metrics_repo: Optional[MetricsRepository] = None,
        companies: Optional[List[Company]] = None,
    ) -> None:
        self.metrics_repo = metrics_repo or MetricsRepository()
        # Build a lookup: company_id -> Company
        self._company_map: Dict[str, Company] = {
            c.id: c for c in (companies or [])
        }

    def identify_underperformers(self, sector: str) -> List[UnderperformerResult]:
        """Identify companies whose Innovation Score is below the sector average.

        Args:
            sector: Industry sector name (e.g. 'Technology').

        Returns:
            List of UnderperformerResult objects sorted by gap descending.

        Validates: Requirement 9.1
        """
        all_records: List[InnovationScoreRecord] = self.metrics_repo.get_by_type(
            InnovationScoreRecord
        )  # type: ignore[assignment]

        sector_records: List[InnovationScoreRecord] = []
        for record in all_records:
            company = self._company_map.get(record.company_id)
            if company is not None and company.sector == sector:
                sector_records.append(record)

        if not sector_records:
            logger.debug("No Innovation Score records found for sector '%s'", sector)
            return []

        latest: Dict[str, InnovationScoreRecord] = {}
        for record in sector_records:
            existing = latest.get(record.company_id)
            if existing is None or record.timestamp > existing.timestamp:
                latest[record.company_id] = record

        scores = {cid: rec.normalized_score for cid, rec in latest.items()}
        sector_average = sum(scores.values()) / len(scores)

        results: List[UnderperformerResult] = []
        for company_id, score in scores.items():
            if score < sector_average:
                company = self._company_map[company_id]
                gap = sector_average - score
                results.append(
                    UnderperformerResult(
                        company=company,
                        innovation_score=score,
                        sector_average=sector_average,
                        gap=gap,
                    )
                )

        results.sort(key=lambda r: r.gap, reverse=True)
        return results

    def recommend_investments(self, quartile: str = "bottom") -> List[Recommendation]:
        """Generate investment recommendations for companies in specified quartile.

        Args:
            quartile: Target quartile ('bottom', 'top', 'upper_mid', 'lower_mid').

        Returns:
            List of Recommendation objects with strategy, expected_outcome, confidence.

        Validates: Requirements 9.2, 9.4
        """
        all_records: List[DigitalMaturityRecord] = self.metrics_repo.get_by_type(
            DigitalMaturityRecord
        )  # type: ignore[assignment]

        latest: Dict[str, DigitalMaturityRecord] = {}
        for record in all_records:
            if record.quartile != quartile:
                continue
            existing = latest.get(record.company_id)
            if existing is None or record.timestamp > existing.timestamp:
                latest[record.company_id] = record

        if not latest:
            return []

        recommendations: List[Recommendation] = []
        for company_id, record in latest.items():
            company = self._company_map.get(company_id)
            company_name = company.name if company is not None else company_id

            dmi = record.metric_value
            sector_avg = record.sector_avg
            gap = max(sector_avg - dmi, 0.0)

            if sector_avg > 0:
                confidence_level = min(gap / sector_avg, 1.0)
                contribution_increase_pct = round((gap / sector_avg) * 100, 1)
            else:
                confidence_level = 0.5
                contribution_increase_pct = 0.0

            talent_improvement_pct = round(contribution_increase_pct * 0.5, 1)

            expected_outcome = (
                f"Estimated {talent_improvement_pct}% improvement in talent attraction "
                f"based on {contribution_increase_pct}% increase in open-source contributions"
            )

            description = (
                f"Invest in open-source initiatives to raise {company_name}'s Digital "
                f"Maturity Index from {dmi:.2f} toward the {record.sector} sector average "
                f"of {sector_avg:.2f}. Recommended actions include sponsoring key "
                f"open-source projects, increasing GitHub repository activity (stars, "
                f"forks, contributors), and establishing an inner-source programme."
            )

            recommendation = Recommendation(
                priority=1,
                category="investment",
                title=f"Open-Source Investment Strategy for {company_name}",
                description=description,
                target_companies=[company_id],
                expected_outcome=expected_outcome,
                confidence_level=round(confidence_level, 4),
                supporting_metrics={
                    "digital_maturity_index": dmi,
                    "sector_avg": sector_avg,
                    "gap": gap,
                },
            )
            validate_recommendation(recommendation)
            recommendations.append(recommendation)

        return recommendations

    def identify_acquisition_targets(self) -> List[AcquisitionTarget]:
        """Identify companies with high Ecosystem Centrality and low market valuation.

        Returns:
            List of AcquisitionTarget objects sorted by betweenness_centrality descending.

        Validates: Requirement 9.3
        """
        all_records: List[EcosystemCentralityRecord] = self.metrics_repo.get_by_type(
            EcosystemCentralityRecord
        )  # type: ignore[assignment]

        if not all_records:
            return []

        latest: Dict[str, EcosystemCentralityRecord] = {}
        for record in all_records:
            existing = latest.get(record.company_id)
            if existing is None or record.timestamp > existing.timestamp:
                latest[record.company_id] = record

        sector_groups: Dict[str, List[EcosystemCentralityRecord]] = {}
        for record in latest.values():
            company = self._company_map.get(record.company_id)
            sector = company.sector if company is not None else "Unknown"
            sector_groups.setdefault(sector, []).append(record)

        def _median(values: List[float]) -> float:
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            mid = n // 2
            if n % 2 == 1:
                return sorted_vals[mid]
            return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0

        targets: List[AcquisitionTarget] = []

        for sector, records in sector_groups.items():
            centralities = [r.betweenness_centrality for r in records]
            median_centrality = _median(centralities)

            revenue_ranks = []
            for r in records:
                company = self._company_map.get(r.company_id)
                if company is not None:
                    revenue_ranks.append(float(company.revenue_rank))
            median_revenue_rank = _median(revenue_ranks) if revenue_ranks else 0.0

            for record in records:
                company = self._company_map.get(record.company_id)
                if company is None:
                    continue

                high_centrality = record.betweenness_centrality > median_centrality
                low_valuation = float(company.revenue_rank) > median_revenue_rank

                if high_centrality and low_valuation:
                    rationale = (
                        f"{company.name} is a strong acquisition candidate in the "
                        f"{sector} sector: its betweenness centrality "
                        f"({record.betweenness_centrality:.4f}) exceeds the sector "
                        f"median ({median_centrality:.4f}), while its revenue rank "
                        f"({company.revenue_rank}) is above the sector median "
                        f"({median_revenue_rank:.1f}), suggesting lower market valuation."
                    )
                    metrics: Dict[str, float] = {
                        "ecosystem_centrality": record.betweenness_centrality,
                        "revenue_rank": float(company.revenue_rank),
                        "sector_median_centrality": median_centrality,
                        "sector_median_revenue_rank": median_revenue_rank,
                    }
                    targets.append(
                        AcquisitionTarget(
                            company_id=company.id,
                            company_name=company.name,
                            rationale=rationale,
                            metrics=metrics,
                        )
                    )

        targets.sort(key=lambda t: t.metrics["ecosystem_centrality"], reverse=True)
        return targets

    def calculate_roi(
        self,
        traditional_hours: float = 2000.0,
        system_hours: float = 200.0,
        hourly_rate: float = 150.0,
        old_decision_time: float = 14.0,
        new_decision_time: float = 2.0,
        turnover_rate: float = 0.10,
        knowledge_base_value: float = 500_000.0,
        system_costs: float = 100_000.0,
    ) -> ROIMetrics:
        """Calculate system ROI based on time savings and decision impact.

        Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5
        """
        time_savings = (traditional_hours - system_hours) * hourly_rate
        revenue_impact = self._calculate_revenue_impact()

        if old_decision_time > 0:
            decision_speed_improvement = (
                (old_decision_time - new_decision_time) / old_decision_time
            ) * 100.0
        else:
            decision_speed_improvement = 0.0

        knowledge_loss_avoidance = turnover_rate * knowledge_base_value
        total_benefits = time_savings + revenue_impact + knowledge_loss_avoidance
        roi_ratio = total_benefits / system_costs if system_costs > 0 else 0.0

        return ROIMetrics(
            time_savings=time_savings,
            revenue_impact=revenue_impact,
            decision_speed_improvement=decision_speed_improvement,
            knowledge_loss_avoidance=knowledge_loss_avoidance,
            total_benefits=total_benefits,
            system_costs=system_costs,
            roi_ratio=roi_ratio,
        )

    def _calculate_revenue_impact(self) -> float:
        """Calculate revenue impact as top quartile avg minus bottom quartile avg."""
        all_records: List[InnovationScoreRecord] = self.metrics_repo.get_by_type(
            InnovationScoreRecord
        )  # type: ignore[assignment]

        if not all_records:
            return 0.0

        latest: Dict[str, InnovationScoreRecord] = {}
        for record in all_records:
            existing = latest.get(record.company_id)
            if existing is None or record.timestamp > existing.timestamp:
                latest[record.company_id] = record

        company_revenues: List[float] = []
        for company_id in latest:
            company = self._company_map.get(company_id)
            if company is not None:
                company_revenues.append(float(company.revenue_rank))

        if len(company_revenues) < 4:
            return 0.0

        company_revenues.sort()
        n = len(company_revenues)
        quartile_size = n // 4

        bottom_quartile = company_revenues[:quartile_size]
        top_quartile = company_revenues[n - quartile_size:]

        bottom_avg = sum(bottom_quartile) / len(bottom_quartile)
        top_avg = sum(top_quartile) / len(top_quartile)

        return bottom_avg - top_avg

    def generate_executive_report(self) -> ExecutiveReport:
        """Produce comprehensive executive report with metrics and recommendations.

        Returns:
            ExecutiveReport object with all sections populated.
            pdf_path and html_path are set to None (use export_executive_report to export).

        Validates: Requirements 11.1, 11.2, 11.3, 11.4, 11.5
        """
        generation_date = datetime.now()
        report_id = str(uuid.uuid4())
        quarter = (generation_date.month - 1) // 3 + 1
        time_period = f"Q{quarter} {generation_date.year}"

        metrics_summary = self._build_metrics_summary()
        leaderboard = self._build_leaderboard()
        trends = self._build_trends_analysis()
        recommendations = self._build_recommendations()
        roi_analysis = self._build_roi_analysis()

        report = ExecutiveReport(
            report_id=report_id,
            generation_date=generation_date,
            time_period=time_period,
            metrics_summary=metrics_summary,
            leaderboard=leaderboard,
            trends=trends,
            recommendations=recommendations,
            roi_analysis=roi_analysis,
            pdf_path=None,
            html_path=None,
        )

        logger.info(
            "Executive report generated: id=%s period=%s companies=%d recommendations=%d",
            report_id,
            time_period,
            metrics_summary.total_companies,
            len(recommendations),
        )
        return report

    def _build_metrics_summary(self) -> MetricsSummary:
        """Build MetricsSummary by aggregating metrics from the MetricsRepository."""
        all_innovation: List[InnovationScoreRecord] = self.metrics_repo.get_by_type(
            InnovationScoreRecord
        )  # type: ignore[assignment]
        latest_innovation: Dict[str, InnovationScoreRecord] = {}
        for record in all_innovation:
            existing = latest_innovation.get(record.company_id)
            if existing is None or record.timestamp > existing.timestamp:
                latest_innovation[record.company_id] = record

        all_dmi: List[DigitalMaturityRecord] = self.metrics_repo.get_by_type(
            DigitalMaturityRecord
        )  # type: ignore[assignment]
        latest_dmi: Dict[str, DigitalMaturityRecord] = {}
        for record in all_dmi:
            existing = latest_dmi.get(record.company_id)
            if existing is None or record.timestamp > existing.timestamp:
                latest_dmi[record.company_id] = record

        total_companies = len(self._company_map)

        if latest_innovation:
            avg_innovation_score = sum(
                r.normalized_score for r in latest_innovation.values()
            ) / len(latest_innovation)
        else:
            avg_innovation_score = 0.0

        if latest_dmi:
            avg_digital_maturity = sum(
                r.metric_value for r in latest_dmi.values()
            ) / len(latest_dmi)
        else:
            avg_digital_maturity = 0.0

        sector_scores: Dict[str, List[float]] = {}
        for company_id, record in latest_innovation.items():
            company = self._company_map.get(company_id)
            if company is not None:
                sector_scores.setdefault(company.sector, []).append(record.normalized_score)
        if sector_scores:
            top_sector = max(
                sector_scores,
                key=lambda s: sum(sector_scores[s]) / len(sector_scores[s]),
            )
        else:
            top_sector = ""

        if latest_innovation:
            best_id = max(latest_innovation, key=lambda cid: latest_innovation[cid].normalized_score)
            best_company = self._company_map.get(best_id)
            highest_growth_company = best_company.name if best_company else best_id
            highest_growth_rate = latest_innovation[best_id].normalized_score
        else:
            highest_growth_company = ""
            highest_growth_rate = 0.0

        return MetricsSummary(
            total_companies=total_companies,
            avg_innovation_score=avg_innovation_score,
            avg_digital_maturity=avg_digital_maturity,
            top_sector=top_sector,
            highest_growth_company=highest_growth_company,
            highest_growth_rate=highest_growth_rate,
        )

    def _build_leaderboard(self) -> List[LeaderboardEntry]:
        """Build leaderboard by ranking companies by Innovation Score (descending)."""
        all_innovation: List[InnovationScoreRecord] = self.metrics_repo.get_by_type(
            InnovationScoreRecord
        )  # type: ignore[assignment]
        latest_innovation: Dict[str, InnovationScoreRecord] = {}
        for record in all_innovation:
            existing = latest_innovation.get(record.company_id)
            if existing is None or record.timestamp > existing.timestamp:
                latest_innovation[record.company_id] = record

        all_dmi: List[DigitalMaturityRecord] = self.metrics_repo.get_by_type(
            DigitalMaturityRecord
        )  # type: ignore[assignment]
        latest_dmi: Dict[str, DigitalMaturityRecord] = {}
        for record in all_dmi:
            existing = latest_dmi.get(record.company_id)
            if existing is None or record.timestamp > existing.timestamp:
                latest_dmi[record.company_id] = record

        all_centrality: List[EcosystemCentralityRecord] = self.metrics_repo.get_by_type(
            EcosystemCentralityRecord
        )  # type: ignore[assignment]
        latest_centrality: Dict[str, EcosystemCentralityRecord] = {}
        for record in all_centrality:
            existing = latest_centrality.get(record.company_id)
            if existing is None or record.timestamp > existing.timestamp:
                latest_centrality[record.company_id] = record

        # Compute year-over-year change per company
        yoy_changes: Dict[str, float] = {}
        all_by_company: Dict[str, List[InnovationScoreRecord]] = {}
        for record in all_innovation:
            all_by_company.setdefault(record.company_id, []).append(record)
        for company_id, records in all_by_company.items():
            records_sorted = sorted(records, key=lambda r: r.timestamp)
            if len(records_sorted) >= 2:
                prev_score = records_sorted[-2].normalized_score
                curr_score = records_sorted[-1].normalized_score
                if prev_score != 0:
                    yoy_changes[company_id] = (curr_score - prev_score) / prev_score
                else:
                    yoy_changes[company_id] = 0.0
            else:
                yoy_changes[company_id] = 0.0

        sorted_companies = sorted(
            latest_innovation.keys(),
            key=lambda cid: latest_innovation[cid].normalized_score,
            reverse=True,
        )

        entries: List[LeaderboardEntry] = []
        for rank, company_id in enumerate(sorted_companies, start=1):
            record = latest_innovation[company_id]
            company = self._company_map.get(company_id)
            company_name = company.name if company is not None else company_id
            sector = company.sector if company is not None else ""

            dmi_record = latest_dmi.get(company_id)
            digital_maturity = dmi_record.metric_value if dmi_record is not None else 0.0

            centrality_record = latest_centrality.get(company_id)
            ecosystem_centrality = (
                centrality_record.betweenness_centrality
                if centrality_record is not None
                else 0.0
            )

            entries.append(
                LeaderboardEntry(
                    rank=rank,
                    company_name=company_name,
                    sector=sector,
                    innovation_score=float(record.normalized_score),
                    digital_maturity=digital_maturity,
                    ecosystem_centrality=ecosystem_centrality,
                    yoy_change=yoy_changes.get(company_id, 0.0),
                )
            )

        return entries

    def _build_trends_analysis(self) -> TrendsAnalysis:
        """Build TrendsAnalysis from historical metric records."""
        all_innovation: List[InnovationScoreRecord] = self.metrics_repo.get_by_type(
            InnovationScoreRecord
        )  # type: ignore[assignment]

        # Aggregate average Innovation Score per date (truncated to day)
        date_scores: Dict[datetime, List[float]] = {}
        for record in all_innovation:
            day = record.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            date_scores.setdefault(day, []).append(record.normalized_score)

        innovation_score_trend: List[Tuple[datetime, float]] = sorted(
            [
                (day, sum(scores) / len(scores))
                for day, scores in date_scores.items()
            ],
            key=lambda x: x[0],
        )

        all_dmi: List[DigitalMaturityRecord] = self.metrics_repo.get_by_type(
            DigitalMaturityRecord
        )  # type: ignore[assignment]

        date_dmi: Dict[datetime, List[float]] = {}
        for record in all_dmi:
            day = record.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            date_dmi.setdefault(day, []).append(record.metric_value)

        digital_maturity_trend: List[Tuple[datetime, float]] = sorted(
            [
                (day, sum(vals) / len(vals))
                for day, vals in date_dmi.items()
            ],
            key=lambda x: x[0],
        )

        # Sector trends: per-sector average Innovation Score per date
        sector_date_scores: Dict[str, Dict[datetime, List[float]]] = {}
        for record in all_innovation:
            company = self._company_map.get(record.company_id)
            sector = company.sector if company is not None else "Unknown"
            day = record.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            sector_date_scores.setdefault(sector, {}).setdefault(day, []).append(
                record.normalized_score
            )

        sector_trends: Dict[str, List[Tuple[datetime, float]]] = {}
        for sector, date_map in sector_date_scores.items():
            sector_trends[sector] = sorted(
                [
                    (day, sum(scores) / len(scores))
                    for day, scores in date_map.items()
                ],
                key=lambda x: x[0],
            )

        inflection_points = self._detect_inflection_points(innovation_score_trend)

        return TrendsAnalysis(
            innovation_score_trend=innovation_score_trend,
            digital_maturity_trend=digital_maturity_trend,
            sector_trends=sector_trends,
            inflection_points=inflection_points,
        )

    def _detect_inflection_points(
        self, trend: List[Tuple[datetime, float]]
    ) -> List[InflectionPoint]:
        """Detect inflection points where trend changes direction."""
        if len(trend) < 3:
            return []

        inflection_points: List[InflectionPoint] = []
        for i in range(1, len(trend) - 1):
            prev_val = trend[i - 1][1]
            curr_val = trend[i][1]
            next_val = trend[i + 1][1]

            was_increasing = curr_val > prev_val
            is_increasing = next_val > curr_val

            if was_increasing and not is_increasing:
                direction_change = "increasing_to_decreasing"
            elif not was_increasing and is_increasing:
                direction_change = "decreasing_to_increasing"
            else:
                continue

            inflection_points.append(
                InflectionPoint(
                    timestamp=trend[i][0],
                    metric_name="innovation_score",
                    value=curr_val,
                    direction_change=direction_change,
                )
            )

        return inflection_points

    def _build_recommendations(self) -> List[Recommendation]:
        """Build prioritized recommendations from existing insight methods."""
        recommendations: List[Recommendation] = []

        # Investment recommendations for bottom quartile (priority 1)
        investment_recs = self.recommend_investments(quartile="bottom")
        for rec in investment_recs:
            recommendations.append(rec)

        # Acquisition target recommendations (priority 2)
        acquisition_targets = self.identify_acquisition_targets()
        for target in acquisition_targets:
            rec = Recommendation(
                priority=2,
                category="acquisition",
                title=f"Acquisition Opportunity: {target.company_name}",
                description=target.rationale,
                target_companies=[target.company_id],
                expected_outcome=(
                    "Strengthen network position and acquire high-centrality assets "
                    "at below-median market valuation."
                ),
                confidence_level=0.75,
                supporting_metrics=target.metrics,
            )
            recommendations.append(rec)

        # Underperformer recommendations per sector (priority 3)
        sectors = {c.sector for c in self._company_map.values()}
        for sector in sorted(sectors):
            underperformers = self.identify_underperformers(sector)
            for result in underperformers:
                rec = Recommendation(
                    priority=3,
                    category="investment",
                    title=f"Innovation Improvement for {result.company.name}",
                    description=(
                        f"{result.company.name} has an Innovation Score of "
                        f"{result.innovation_score:.2f}, which is {result.gap:.2f} "
                        f"below the {sector} sector average of {result.sector_average:.2f}."
                    ),
                    target_companies=[result.company.id],
                    expected_outcome=(
                        f"Close the {result.gap:.2f} gap to sector average through "
                        f"targeted open-source and R&D investments."
                    ),
                    confidence_level=min(result.gap / max(result.sector_average, 1.0), 1.0),
                    supporting_metrics={
                        "innovation_score": result.innovation_score,
                        "sector_average": result.sector_average,
                        "gap": result.gap,
                    },
                )
                recommendations.append(rec)

        recommendations.sort(key=lambda r: r.priority)
        return recommendations

    def _build_roi_analysis(self) -> ROIAnalysis:
        """Build ROIAnalysis by calling calculate_roi() and mapping to ROIAnalysis."""
        roi_metrics = self.calculate_roi()

        all_innovation: List[InnovationScoreRecord] = self.metrics_repo.get_by_type(
            InnovationScoreRecord
        )  # type: ignore[assignment]
        latest_innovation: Dict[str, InnovationScoreRecord] = {}
        for record in all_innovation:
            existing = latest_innovation.get(record.company_id)
            if existing is None or record.timestamp > existing.timestamp:
                latest_innovation[record.company_id] = record

        revenue_ranks: List[float] = []
        for company_id in latest_innovation:
            company = self._company_map.get(company_id)
            if company is not None:
                revenue_ranks.append(float(company.revenue_rank))

        if len(revenue_ranks) >= 4:
            revenue_ranks_sorted = sorted(revenue_ranks)
            n = len(revenue_ranks_sorted)
            q_size = n // 4
            top_quartile_avg = sum(revenue_ranks_sorted[:q_size]) / q_size
            bottom_quartile_avg = sum(revenue_ranks_sorted[n - q_size:]) / q_size
        else:
            top_quartile_avg = 0.0
            bottom_quartile_avg = 0.0

        return ROIAnalysis(
            time_savings_hours=roi_metrics.time_savings / max(roi_metrics.system_costs / 100_000, 1.0) * 1800.0 / 150.0,
            time_savings_value=roi_metrics.time_savings,
            revenue_impact_top_quartile=top_quartile_avg,
            revenue_impact_bottom_quartile=bottom_quartile_avg,
            decision_speed_improvement=roi_metrics.decision_speed_improvement,
            knowledge_loss_avoidance=roi_metrics.knowledge_loss_avoidance,
            total_benefits=roi_metrics.total_benefits,
            system_costs=roi_metrics.system_costs,
            roi_ratio=roi_metrics.roi_ratio,
        )

    # ------------------------------------------------------------------
    # Task 13.3 - Report export (Requirement 11.6)
    # ------------------------------------------------------------------

    def export_executive_report(
        self, report: ExecutiveReport, output_dir: str
    ) -> ExecutiveReport:
        """Export an ExecutiveReport to PDF and interactive HTML formats.

        Generates PDF and HTML files in output_dir.

        Args:
            report: The ExecutiveReport to export.
            output_dir: Directory where the exported files will be saved.

        Returns:
            A copy of the report with pdf_path and html_path populated.

        Validates: Requirement 11.6
        """
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.exception("Failed to create output directory: %s", output_dir)
            raise

        pdf_path = self._export_pdf(report, output_dir)
        html_path = self._export_html(report, output_dir)

        from dataclasses import replace as dc_replace
        updated = dc_replace(report, pdf_path=pdf_path, html_path=html_path)
        logger.info(
            "Executive report exported: id=%s pdf=%s html=%s",
            report.report_id, pdf_path, html_path,
        )
        return updated

    def _export_pdf(self, report: ExecutiveReport, output_dir: str) -> str:
        """Generate a PDF file for the report using ReportLab."""
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib import colors
            from reportlab.platypus import (
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            )
        except ImportError as exc:
            logger.error("ReportLab is not installed: %s", exc)
            raise

        filename = f"executive_report_{report.report_id}.pdf"
        pdf_path = os.path.join(output_dir, filename)

        try:
            doc = SimpleDocTemplate(pdf_path, pagesize=letter)
            styles = getSampleStyleSheet()
            story = []

            title_style = ParagraphStyle(
                "CustomTitle", parent=styles["Title"], fontSize=18, spaceAfter=12,
            )
            heading_style = ParagraphStyle(
                "CustomHeading", parent=styles["Heading1"], fontSize=14, spaceAfter=8,
            )
            normal = styles["Normal"]

            story.append(Paragraph("Executive Report", title_style))
            story.append(Paragraph(
                f"Period: {report.time_period} | Generated: "
                f"{report.generation_date.strftime('%Y-%m-%d %H:%M')}",
                normal,
            ))
            story.append(Spacer(1, 0.2 * inch))

            # Metrics Summary
            story.append(Paragraph("Metrics Summary", heading_style))
            ms = report.metrics_summary
            summary_data = [
                ["Metric", "Value"],
                ["Total Companies", str(ms.total_companies)],
                ["Avg Innovation Score", f"{ms.avg_innovation_score:.2f}"],
                ["Avg Digital Maturity", f"{ms.avg_digital_maturity:.2f}"],
                ["Top Sector", ms.top_sector],
                ["Highest Growth Company", ms.highest_growth_company],
                ["Highest Growth Rate", f"{ms.highest_growth_rate:.2%}"],
            ]
            t = Table(summary_data, colWidths=[3 * inch, 3 * inch])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ]))
            story.append(t)
            story.append(Spacer(1, 0.2 * inch))

            # Leaderboard
            story.append(Paragraph("Leaderboard (Top Companies by Innovation Score)", heading_style))
            top_entries = report.leaderboard[:10]
            if top_entries:
                lb_data = [["Rank", "Company", "Sector", "Innovation Score", "YoY Change"]]
                for entry in top_entries:
                    lb_data.append([
                        str(entry.rank), entry.company_name, entry.sector,
                        f"{entry.innovation_score:.2f}", f"{entry.yoy_change:+.2%}",
                    ])
                lb = Table(lb_data, colWidths=[0.6*inch, 2.2*inch, 1.5*inch, 1.4*inch, 1.0*inch])
                lb.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightblue]),
                ]))
                story.append(lb)
            else:
                story.append(Paragraph("No leaderboard data available.", normal))
            story.append(Spacer(1, 0.2 * inch))

            # Trends
            story.append(Paragraph("Trends Analysis", heading_style))
            if report.trends.innovation_score_trend:
                story.append(Paragraph("Innovation Score Trend:", normal))
                trend_data = [["Date", "Avg Innovation Score"]]
                for ts, val in report.trends.innovation_score_trend:
                    trend_data.append([ts.strftime("%Y-%m-%d"), f"{val:.2f}"])
                tt = Table(trend_data, colWidths=[2 * inch, 2 * inch])
                tt.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ]))
                story.append(tt)
            else:
                story.append(Paragraph("No trend data available.", normal))
            story.append(Spacer(1, 0.2 * inch))

            # Recommendations
            story.append(Paragraph("Recommendations", heading_style))
            if report.recommendations:
                for rec in report.recommendations:
                    story.append(Paragraph(
                        f"<b>[Priority {rec.priority}] {rec.title}</b>", normal,
                    ))
                    story.append(Paragraph(rec.description, normal))
                    story.append(Paragraph(
                        f"Expected outcome: {rec.expected_outcome} "
                        f"(confidence: {rec.confidence_level:.0%})",
                        normal,
                    ))
                    story.append(Spacer(1, 0.1 * inch))
            else:
                story.append(Paragraph("No recommendations available.", normal))
            story.append(Spacer(1, 0.2 * inch))

            # ROI Analysis
            story.append(Paragraph("ROI Analysis", heading_style))
            roi = report.roi_analysis
            roi_data = [
                ["Metric", "Value"],
                ["Time Savings (hours)", f"{roi.time_savings_hours:.1f}"],
                ["Time Savings (value)", f"${roi.time_savings_value:,.0f}"],
                ["Revenue Impact - Top Quartile", f"${roi.revenue_impact_top_quartile:,.0f}"],
                ["Revenue Impact - Bottom Quartile", f"${roi.revenue_impact_bottom_quartile:,.0f}"],
                ["Decision Speed Improvement", f"{roi.decision_speed_improvement:.1f}%"],
                ["Knowledge Loss Avoidance", f"${roi.knowledge_loss_avoidance:,.0f}"],
                ["Total Benefits", f"${roi.total_benefits:,.0f}"],
                ["System Costs", f"${roi.system_costs:,.0f}"],
                ["ROI Ratio", f"{roi.roi_ratio:.2f}x"],
            ]
            rt = Table(roi_data, colWidths=[3 * inch, 3 * inch])
            rt.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ]))
            story.append(rt)

            doc.build(story)
        except Exception:
            logger.exception("Failed to generate PDF report: %s", pdf_path)
            raise

        return pdf_path

    def _export_html(self, report: ExecutiveReport, output_dir: str) -> str:
        """Generate an interactive HTML file for the report using Jinja2."""
        try:
            from jinja2 import Environment, FileSystemLoader, select_autoescape
        except ImportError as exc:
            logger.error("Jinja2 is not installed: %s", exc)
            raise

        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(["html"]),
        )

        filename = f"executive_report_{report.report_id}.html"
        html_path = os.path.join(output_dir, filename)

        try:
            template = env.get_template("executive_report.html.j2")
            html_content = template.render(report=report)
            with open(html_path, "w", encoding="utf-8") as fh:
                fh.write(html_content)
        except Exception:
            logger.exception("Failed to generate HTML report: %s", html_path)
            raise

        return html_path

    # ------------------------------------------------------------------
    # Best Practice Identification (Requirement 13.5)
    # ------------------------------------------------------------------

    def identify_best_practices(
        self,
        sector_averages: Dict[str, Dict[str, float]],
        metric_names: Optional[List[str]] = None,
    ) -> List[BestPractice]:
        """
        Identify best practices from high-performing sectors.

        A sector is considered "high-performing" for a given metric when its
        average value is above the median across all sectors for that metric.
        Best practices are generated for every metric where at least one sector
        exceeds the median, and are targeted at sectors that fall below the median.

        Args:
            sector_averages: Mapping of sector -> {metric_name -> average_value},
                             as returned by AnalyticsEngine.calculate_sector_averages().
            metric_names: Optional list of metric names to analyse. When None,
                          all metrics present in sector_averages are used.

        Returns:
            List of BestPractice objects, one per (high-performing sector, metric)
            combination. Ordered by metric name then descending sector average.

        Validates: Requirement 13.5
        """
        if not sector_averages:
            return []

        # Determine which metrics to analyse
        all_metrics: set = set()
        for avgs in sector_averages.values():
            all_metrics.update(avgs.keys())
        metrics_to_analyse = list(metric_names or all_metrics)

        best_practices: List[BestPractice] = []

        for metric in sorted(metrics_to_analyse):
            # Collect per-sector averages for this metric
            metric_values: Dict[str, float] = {
                sector: avgs[metric]
                for sector, avgs in sector_averages.items()
                if metric in avgs
            }
            if len(metric_values) < 2:
                # Need at least 2 sectors to identify high vs low performers
                continue

            sorted_values = sorted(metric_values.values())
            n = len(sorted_values)
            # Median calculation
            if n % 2 == 1:
                overall_median = sorted_values[n // 2]
            else:
                overall_median = (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2.0

            # High-performing sectors: above median
            high_performers = [
                sector for sector, avg in metric_values.items()
                if avg > overall_median
            ]
            # Target sectors: at or below median
            target_sectors = [
                sector for sector, avg in metric_values.items()
                if avg <= overall_median
            ]

            for sector in sorted(high_performers, key=lambda s: metric_values[s], reverse=True):
                description = (
                    f"Sector '{sector}' achieves an above-median average "
                    f"{metric} of {metric_values[sector]:.4f} "
                    f"(median: {overall_median:.4f}). "
                    f"Lagging sectors ({', '.join(sorted(target_sectors))}) "
                    f"should adopt practices from this sector to improve their {metric}."
                )
                best_practices.append(
                    BestPractice(
                        sector=sector,
                        metric_name=metric,
                        sector_avg=metric_values[sector],
                        overall_median=overall_median,
                        description=description,
                        target_sectors=sorted(target_sectors),
                    )
                )

        logger.info(
            "Identified %d best practices from %d sectors across %d metrics",
            len(best_practices),
            len(sector_averages),
            len(metrics_to_analyse),
        )
        return best_practices
