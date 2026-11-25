from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, BinaryIO
import base64
import html as html_module
import io
import os
import tempfile

ImageData = Union[bytes, str, Path, 'np.ndarray']

class OutputFormat(Enum):
    HTML = auto()
    PDF = auto()
    MARKDOWN = auto()

class SectionType(Enum):
    TEXT = auto()
    TABLE = auto()
    IMAGE = auto()
    CHART = auto()
    METRICS = auto()
    RECOMMENDATIONS = auto()

@dataclass
class ReportConfig:
    output_format: OutputFormat = OutputFormat.HTML
    title: str = "Casting Simulation Report"
    author: str = ""
    company: str = ""
    logo_path: Optional[Path] = None
    include_toc: bool = True
    page_size: str = "A4"
    include_timestamp: bool = True
    theme: str = "default"

@dataclass
class ReportSection:
    title: str
    section_type: SectionType
    content: Any
    order: int = 0
    subsections: List['ReportSection'] = field(default_factory=list)

@dataclass
class TableData:
    headers: List[str]
    rows: List[List[Any]]
    caption: str = ""
    highlight_rules: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MetricsData:
    metrics: Dict[str, float]
    units: Dict[str, str] = field(default_factory=dict)
    thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)

@dataclass
class RecommendationData:
    recommendations: List[str]
    severity: str = "info"
    category: str = "general"

class ReportGenerator:

    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
        self._sections: List[ReportSection] = []
        self._images: Dict[str, bytes] = {}
        self._order_counter = 0

    def add_text_section(
        self,
        title: str,
        content: str,
        order: Optional[int] = None
    ) -> None:
        self._add_section(title, SectionType.TEXT, content, order)

    def add_table_section(
        self,
        title: str,
        table: TableData,
        order: Optional[int] = None
    ) -> None:
        self._add_section(title, SectionType.TABLE, table, order)

    def add_image_section(
        self,
        title: str,
        image: ImageData,
        caption: str = "",
        order: Optional[int] = None
    ) -> None:
        image_bytes = self._process_image(image)
        image_id = f"img_{len(self._images)}"
        self._images[image_id] = image_bytes

        self._add_section(
            title,
            SectionType.IMAGE,
            {'image_id': image_id, 'caption': caption},
            order
        )

    def add_metrics_section(
        self,
        title: str,
        metrics: MetricsData,
        order: Optional[int] = None
    ) -> None:
        self._add_section(title, SectionType.METRICS, metrics, order)

    def add_recommendations_section(
        self,
        title: str,
        recommendations: RecommendationData,
        order: Optional[int] = None
    ) -> None:
        self._add_section(title, SectionType.RECOMMENDATIONS, recommendations, order)

    def _add_section(
        self,
        title: str,
        section_type: SectionType,
        content: Any,
        order: Optional[int]
    ) -> None:
        if order is None:
            order = self._order_counter
            self._order_counter += 1

        section = ReportSection(
            title=title,
            section_type=section_type,
            content=content,
            order=order
        )
        self._sections.append(section)

    def _process_image(self, image: ImageData) -> bytes:
        if isinstance(image, bytes):
            return image

        if isinstance(image, str):
            image = Path(image)

        if isinstance(image, Path):
            with open(image, 'rb') as f:
                return f.read()

        try:
            import numpy as np
            from PIL import Image

            if isinstance(image, np.ndarray):
                if image.dtype in [np.float32, np.float64]:
                    image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)

                img = Image.fromarray(image)
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                return buffer.getvalue()
        except ImportError:
            pass

        raise ValueError(f"Cannot process image of type {type(image)}")

    def generate(self, output_path: Union[str, Path]) -> Path:
        output_path = Path(output_path)

        self._sections.sort(key=lambda s: s.order)

        if self.config.output_format == OutputFormat.HTML:
            content = self._generate_html()
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)

        elif self.config.output_format == OutputFormat.PDF:
            html_content = self._generate_html()
            self._html_to_pdf(html_content, output_path)

        elif self.config.output_format == OutputFormat.MARKDOWN:
            content = self._generate_markdown()
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)

        return output_path

    @staticmethod
    def _esc(value: Any) -> str:
        return html_module.escape("" if value is None else str(value), quote=True)

    @staticmethod
    def _esc_md(value: Any) -> str:
        text = "" if value is None else str(value)
        for char in ("\\", "`", "*", "_", "[", "]", "(", ")", "#", "<", ">", "|"):
            text = text.replace(char, "\\" + char)
        return text

    def _generate_html(self) -> str:
        html_parts = [self._html_header()]

        for section in self._sections:
            html_parts.append(self._render_section_html(section))

        html_parts.append(self._html_footer())

        return '\n'.join(html_parts)

    def _html_header(self) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if self.config.include_timestamp else ""

        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self._esc(self.config.title)}</title>
    <style>
        {self._get_css_styles()}
    </style>
</head>
<body>
    <div class="report-container">
        <header class="report-header">
            <h1>{self._esc(self.config.title)}</h1>
            {f'<p class="author">Author: {self._esc(self.config.author)}</p>' if self.config.author else ''}
            {f'<p class="company">{self._esc(self.config.company)}</p>' if self.config.company else ''}
            {f'<p class="timestamp">Generated: {self._esc(timestamp)}</p>' if timestamp else ''}
        </header>
        <main class="report-content">
'''

    def _html_footer(self) -> str:
        return '''
        </main>
        <footer class="report-footer">
            <p>Generated by Casting Simulation Software</p>
        </footer>
    </div>
</body>
</html>'''

    def _get_css_styles(self) -> str:
        base_styles = '''
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }
        
        .report-container {
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        
        .report-header {
            background: linear-gradient(135deg, #1a5276, #2980b9);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        .report-header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .report-header .author,
        .report-header .company,
        .report-header .timestamp {
            opacity: 0.9;
            margin: 5px 0;
        }
        
        .report-content {
            padding: 40px;
        }
        
        .section {
            margin-bottom: 40px;
            page-break-inside: avoid;
        }
        
        .section h2 {
            color: #1a5276;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        
        .section-text p {
            margin-bottom: 15px;
            text-align: justify;
        }
        
        .section-table {
            width: 100%;
            overflow-x: auto;
        }
        
        .section-table table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        
        .section-table th,
        .section-table td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .section-table th {
            background-color: #3498db;
            color: white;
            font-weight: 600;
        }
        
        .section-table tr:hover {
            background-color: #f5f5f5;
        }
        
        .section-table caption {
            caption-side: bottom;
            padding: 10px;
            font-style: italic;
            color: #666;
        }
        
        .section-image {
            text-align: center;
        }
        
        .section-image img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            margin: 15px 0;
        }
        
        .section-image figcaption {
            color: #666;
            font-style: italic;
        }
        
        .section-metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 15px;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            border-left: 4px solid #3498db;
        }
        
        .metric-card .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #1a5276;
        }
        
        .metric-card .metric-name {
            color: #666;
            margin-top: 5px;
        }
        
        .metric-card .metric-unit {
            color: #999;
            font-size: 0.9em;
        }
        
        .metric-card.warning {
            border-left-color: #f39c12;
        }
        
        .metric-card.critical {
            border-left-color: #e74c3c;
        }
        
        .section-recommendations ul {
            list-style: none;
            padding: 0;
        }
        
        .section-recommendations li {
            padding: 15px;
            margin: 10px 0;
            background: #f8f9fa;
            border-radius: 4px;
            border-left: 4px solid #3498db;
        }
        
        .section-recommendations li.info {
            border-left-color: #3498db;
        }
        
        .section-recommendations li.warning {
            border-left-color: #f39c12;
            background: #fef9e7;
        }
        
        .section-recommendations li.critical {
            border-left-color: #e74c3c;
            background: #fdedec;
        }
        
        .report-footer {
            background: #34495e;
            color: white;
            padding: 20px;
            text-align: center;
        }
        
        @media print {
            body {
                background: white;
            }
            .report-container {
                box-shadow: none;
            }
            .section {
                page-break-inside: avoid;
            }
        }
        '''

        return base_styles

    def _render_section_html(self, section: ReportSection) -> str:
        if section.section_type == SectionType.TEXT:
            return self._render_text_section(section)
        elif section.section_type == SectionType.TABLE:
            return self._render_table_section(section)
        elif section.section_type == SectionType.IMAGE:
            return self._render_image_section(section)
        elif section.section_type == SectionType.METRICS:
            return self._render_metrics_section(section)
        elif section.section_type == SectionType.RECOMMENDATIONS:
            return self._render_recommendations_section(section)
        else:
            return f'<div class="section"><h2>{section.title}</h2><p>{section.content}</p></div>'

    def _render_text_section(self, section: ReportSection) -> str:
        content = self._esc(section.content)
        content = content.replace('\n\n', '</p><p>').replace('\n', '<br>')
        content = f'<p>{content}</p>'

        return f'''
        <div class="section section-text">
            <h2>{self._esc(section.title)}</h2>
            {content}
        </div>
        '''

    def _render_table_section(self, section: ReportSection) -> str:
        table: TableData = section.content

        header_html = ''.join(f'<th>{self._esc(h)}</th>' for h in table.headers)

        rows_html = ''
        for row in table.rows:
            cells = ''.join(f'<td>{self._esc(cell)}</td>' for cell in row)
            rows_html += f'<tr>{cells}</tr>'

        caption = f'<caption>{self._esc(table.caption)}</caption>' if table.caption else ''

        return f'''
        <div class="section section-table">
            <h2>{self._esc(section.title)}</h2>
            <table>
                {caption}
                <thead><tr>{header_html}</tr></thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        '''

    def _render_image_section(self, section: ReportSection) -> str:
        image_id = section.content['image_id']
        caption = section.content['caption']

        image_bytes = self._images.get(image_id, b'')
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        caption_html = f'<figcaption>{self._esc(caption)}</figcaption>' if caption else ''

        return f'''
        <div class="section section-image">
            <h2>{self._esc(section.title)}</h2>
            <figure>
                <img src="data:image/png;base64,{image_b64}" alt="{self._esc(section.title)}">
                {caption_html}
            </figure>
        </div>
        '''

    def _render_metrics_section(self, section: ReportSection) -> str:
        metrics: MetricsData = section.content

        cards_html = ''
        for name, value in metrics.metrics.items():
            unit = metrics.units.get(name, '')

            severity_class = ''
            if name in metrics.thresholds:
                thresh = metrics.thresholds[name]
                if 'critical' in thresh and value > thresh['critical']:
                    severity_class = 'critical'
                elif 'warning' in thresh and value > thresh['warning']:
                    severity_class = 'warning'

            cards_html += f'''
            <div class="metric-card {severity_class}">
                <div class="metric-value">{value:.2f}</div>
                <div class="metric-name">{self._esc(name)}</div>
                <div class="metric-unit">{self._esc(unit)}</div>
            </div>
            '''

        return f'''
        <div class="section">
            <h2>{self._esc(section.title)}</h2>
            <div class="section-metrics">
                {cards_html}
            </div>
        </div>
        '''

    def _render_recommendations_section(self, section: ReportSection) -> str:
        recs: RecommendationData = section.content

        items_html = ''
        for rec in recs.recommendations:
            items_html += f'<li class="{self._esc(recs.severity)}">{self._esc(rec)}</li>'

        return f'''
        <div class="section section-recommendations">
            <h2>{self._esc(section.title)}</h2>
            <ul>{items_html}</ul>
        </div>
        '''

    def _html_to_pdf(self, html_content: str, output_path: Path) -> None:
        try:
            from weasyprint import HTML
            HTML(string=html_content).write_pdf(output_path)
            return
        except ImportError:
            pass

        try:
            import pdfkit
            pdfkit.from_string(html_content, str(output_path))
            return
        except ImportError:
            pass

        html_path = output_path.with_suffix('.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        note_path = output_path.with_suffix('.txt')
        with open(note_path, 'w') as f:
            f.write(
                "PDF generation requires weasyprint or pdfkit.\n"
                "Install with: pip install weasyprint\n"
                "The report has been saved as HTML instead.\n"
            )

    def _generate_markdown(self) -> str:
        parts = [
            f"# {self._esc_md(self.config.title)}",
            "",
        ]

        if self.config.author:
            parts.append(f"**Author:** {self._esc_md(self.config.author)}")
        if self.config.company:
            parts.append(f"**Company:** {self._esc_md(self.config.company)}")
        if self.config.include_timestamp:
            parts.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        parts.extend(["", "---", ""])

        for section in self._sections:
            parts.append(f"## {self._esc_md(section.title)}")
            parts.append("")

            if section.section_type == SectionType.TEXT:
                parts.append(self._esc_md(section.content))

            elif section.section_type == SectionType.TABLE:
                table: TableData = section.content
                parts.append("| " + " | ".join(self._esc_md(h) for h in table.headers) + " |")
                parts.append("| " + " | ".join(["---"] * len(table.headers)) + " |")
                for row in table.rows:
                    parts.append("| " + " | ".join(self._esc_md(c) for c in row) + " |")

            elif section.section_type == SectionType.METRICS:
                metrics: MetricsData = section.content
                for name, value in metrics.metrics.items():
                    unit = metrics.units.get(name, '')
                    parts.append(f"- **{self._esc_md(name)}:** {value:.2f} {self._esc_md(unit)}")

            elif section.section_type == SectionType.RECOMMENDATIONS:
                recs: RecommendationData = section.content
                for rec in recs.recommendations:
                    parts.append(f"- {self._esc_md(rec)}")

            parts.append("")

        return "\n".join(parts)
