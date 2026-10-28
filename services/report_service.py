"""ReportService: Generates comprehensive traffic audit reports and exports to CSV and PDF.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import csv
from traffic_ai.config import EXPORTS_DIR
from traffic_ai.db.models import TrafficData, Violation, Incident, SignalLog, Report, db


class ReportService:
    """Handles report compilation, CSV export, and PDF generation."""

    def __init__(self, exports_dir: Optional[Path] = None):
        self.exports_dir = exports_dir or EXPORTS_DIR
        self.exports_dir.mkdir(parents=True, exist_ok=True)

    def generate_report_data(
        self,
        report_type: str = "DAILY_SUMMARY",
        days: int = 1,
        intersection_id: str = "INT_01"
    ) -> Dict[str, Any]:
        """Aggregate statistical data for a given timeframe."""
        start_date = datetime.utcnow() - timedelta(days=days)

        traffic_records = TrafficData.query.filter(
            TrafficData.timestamp >= start_date,
            TrafficData.intersection_id == intersection_id
        ).order_by(TrafficData.timestamp.asc()).all()

        violations = Violation.query.filter(
            Violation.timestamp >= start_date,
            Violation.intersection_id == intersection_id
        ).order_by(Violation.timestamp.desc()).all()

        incidents = Incident.query.filter(
            Incident.timestamp >= start_date,
            Incident.intersection_id == intersection_id
        ).order_by(Incident.timestamp.desc()).all()

        signal_logs = SignalLog.query.filter(
            SignalLog.timestamp >= start_date,
            SignalLog.intersection_id == intersection_id
        ).order_by(SignalLog.timestamp.desc()).all()

        total_vehicles = sum(r.vehicle_count for r in traffic_records)
        avg_density = (sum(r.density for r in traffic_records) / len(traffic_records)) if traffic_records else 0.0
        avg_wait = (sum(s.avg_wait_time for s in signal_logs) / len(signal_logs)) if signal_logs else 0.0

        dir_breakdown: Dict[str, int] = {}
        for r in traffic_records:
            dir_breakdown[r.direction] = dir_breakdown.get(r.direction, 0) + r.vehicle_count

        viol_breakdown: Dict[str, int] = {}
        for v in violations:
            viol_breakdown[v.violation_type] = viol_breakdown.get(v.violation_type, 0) + 1

        return {
            "title": f"Traffic Intelligence Audit: {report_type.replace('_', ' ')}",
            "report_type": report_type,
            "intersection_id": intersection_id,
            "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            "period_days": days,
            "start_date": start_date.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": {
                "total_vehicles": total_vehicles,
                "avg_density": round(avg_density, 3),
                "avg_wait_time_sec": round(avg_wait, 1),
                "total_violations": len(violations),
                "total_incidents": len(incidents),
                "signal_cycles": len(signal_logs)
            },
            "directional_breakdown": dir_breakdown,
            "violation_breakdown": viol_breakdown,
            "recent_traffic": [r.to_dict() for r in traffic_records[-200:]],
            "violations_list": [v.to_dict() for v in violations],
            "incidents_list": [i.to_dict() for i in incidents]
        }

    def export_csv(self, report_data: Dict[str, Any], filename_prefix: str = "traffic_report") -> str:
        """Export tabular traffic records to CSV format."""
        ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{ts_str}.csv"
        file_path = self.exports_dir / filename

        with open(file_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Header info
            writer.writerow(["# AI Traffic Management System Export"])
            writer.writerow(["# Title", report_data.get("title", "")])
            writer.writerow(["# Generated At", report_data.get("generated_at", "")])
            writer.writerow(["# Intersection", report_data.get("intersection_id", "")])
            writer.writerow([])
            # Summary Metrics
            writer.writerow(["Summary Metric", "Value"])
            for k, v in report_data.get("metrics", {}).items():
                writer.writerow([k, v])
            writer.writerow([])
            # Traffic records
            writer.writerow(["Timestamp", "Intersection", "Direction", "Vehicle Count", "Density", "Level", "Avg Speed", "Queue Length"])
            for r in report_data.get("recent_traffic", []):
                writer.writerow([
                    r.get("timestamp"),
                    r.get("intersection_id"),
                    r.get("direction"),
                    r.get("vehicle_count"),
                    r.get("density"),
                    r.get("density_level"),
                    r.get("avg_speed"),
                    r.get("queue_length")
                ])
            writer.writerow([])
            # Violations
            writer.writerow(["Violation ID", "Timestamp", "Type", "Vehicle Type", "Status", "Notes"])
            for v in report_data.get("violations_list", []):
                writer.writerow([
                    v.get("id"),
                    v.get("timestamp"),
                    v.get("violation_type"),
                    v.get("vehicle_type"),
                    v.get("status"),
                    v.get("notes")
                ])

        return str(file_path)

    def export_pdf(self, report_data: Dict[str, Any], filename_prefix: str = "traffic_audit") -> str:
        """Generate PDF report with ReportLab."""
        ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{ts_str}.pdf"
        file_path = self.exports_dir / filename

        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

            doc = SimpleDocTemplate(
                str(file_path),
                pagesize=letter,
                rightMargin=36,
                leftMargin=36,
                topMargin=36,
                bottomMargin=36
            )

            styles = getSampleStyleSheet()
            title_style = ParagraphStyle(
                "TitleStyle",
                parent=styles["Heading1"],
                fontSize=20,
                leading=24,
                textColor=colors.HexColor("#0f172a")
            )
            subtitle_style = ParagraphStyle(
                "SubTitleStyle",
                parent=styles["Normal"],
                fontSize=10,
                textColor=colors.HexColor("#64748b")
            )
            section_style = ParagraphStyle(
                "SectionStyle",
                parent=styles["Heading2"],
                fontSize=13,
                leading=16,
                textColor=colors.HexColor("#1e293b"),
                spaceAfter=6
            )

            elements = []

            # Header
            elements.append(Paragraph(report_data.get("title", "Traffic Report"), title_style))
            elements.append(Paragraph(f"Generated on {report_data.get('generated_at')} | Intersection: {report_data.get('intersection_id')}", subtitle_style))
            elements.append(Spacer(1, 14))

            # Key Metrics Table
            elements.append(Paragraph("Executive Summary Metrics", section_style))
            m = report_data.get("metrics", {})
            summary_data = [
                ["Total Vehicles Processed", str(m.get("total_vehicles", 0)), "Average Wait Time", f"{m.get('avg_wait_time_sec', 0)} s"],
                ["Average Congestion Density", str(m.get("avg_density", 0)), "Total Violations Detected", str(m.get("total_violations", 0))],
                ["Reported Road Incidents", str(m.get("total_incidents", 0)), "Adaptive Signal Cycles", str(m.get("signal_cycles", 0))]
            ]
            t_summary = Table(summary_data, colWidths=[150, 100, 150, 100])
            t_summary.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0f172a")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
                ("FONTNAME", (3, 0), (3, -1), "Helvetica-Bold"),
            ]))
            elements.append(t_summary)
            elements.append(Spacer(1, 14))

            # Directional Volume
            elements.append(Paragraph("Directional Vehicle Distribution", section_style))
            dir_data = [["Direction / Approach", "Vehicle Count", "Proportion"]]
            total_v = max(1, m.get("total_vehicles", 1))
            for d, c in report_data.get("directional_breakdown", {}).items():
                pct = f"{(c / total_v) * 100:.1f}%"
                dir_data.append([d, str(c), pct])
            t_dir = Table(dir_data, colWidths=[180, 160, 160])
            t_dir.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
            ]))
            elements.append(t_dir)
            elements.append(Spacer(1, 14))

            # Violations Table (Top 10)
            elements.append(Paragraph("Recent Traffic Infractions (Operator Audit)", section_style))
            v_rows = [["ID", "Timestamp", "Infraction Type", "Vehicle", "Status"]]
            for v in report_data.get("violations_list", [])[:10]:
                v_rows.append([
                    str(v.get("id")),
                    str(v.get("timestamp"))[:19],
                    str(v.get("violation_type")),
                    str(v.get("vehicle_type")),
                    str(v.get("status"))
                ])
            if len(v_rows) == 1:
                v_rows.append(["-", "No violations recorded", "-", "-", "-"])

            t_v = Table(v_rows, colWidths=[40, 130, 140, 80, 110])
            t_v.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ]))
            elements.append(t_v)

            doc.build(elements)
            return str(file_path)

        except Exception as e:
            # Fallback text file renamed to .pdf or simple writer if error
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"=== {report_data.get('title')} ===\n")
                f.write(f"Generated: {report_data.get('generated_at')}\n\n")
                f.write(f"Metrics: {report_data.get('metrics')}\n")
            return str(file_path)
