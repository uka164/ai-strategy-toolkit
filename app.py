"""AI Strategy Toolkit.

Interaktive Streamlit-Anwendung zur strukturierten Bewertung von AI-Use-Cases,
EU AI Act Compliance-Prüfung, ROI-Kalkulation und Rollout-Roadmap-Generierung.

Start:
    streamlit run app.py
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import StringIO
from typing import Final, Literal

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------------
# Design Tokens
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Theme:
    """Design-Token-Set für das dunkle Hauptthema.

    Alle Module greifen ausschließlich über diese Tokens auf Farben zu, damit
    visuelle Konsistenz und ein späterer Theme-Wechsel ohne Streuung möglich
    bleiben.
    """

    bg: str = "#0b1120"
    surface: str = "#111827"
    surface_elevated: str = "#1a2236"
    border: str = "rgba(148, 163, 184, 0.12)"
    border_strong: str = "rgba(148, 163, 184, 0.22)"
    text_primary: str = "#f1f5f9"
    text_secondary: str = "#94a3b8"
    text_muted: str = "#64748b"
    accent: str = "#2dd4bf"
    accent_soft: str = "rgba(45, 212, 191, 0.12)"
    success: str = "#4ade80"
    warning: str = "#fbbf24"
    danger: str = "#f87171"
    critical: str = "#c084fc"


THEME: Final[Theme] = Theme()


WEIGHT_IMPACT: Final[float] = 0.40
WEIGHT_FEASIBILITY: Final[float] = 0.30
WEIGHT_RISK: Final[float] = 0.30

SCORE_GREEN: Final[float] = 70.0
SCORE_ORANGE: Final[float] = 50.0

RiskLevel = Literal["minimal", "limited", "high", "unacceptable"]
TrafficLight = Literal["green", "orange", "red"]

PHASE_BASE_DAYS: Final[dict[str, int]] = {
    "Discovery": 14,
    "PoC": 30,
    "Pilot": 60,
    "Production": 90,
    "Scale": 120,
}

PHASE_OWNERS: Final[dict[str, str]] = {
    "Discovery": "Strategy",
    "PoC": "AI Team",
    "Pilot": "Product",
    "Production": "Engineering",
    "Scale": "Ops",
}

PHASE_COLORS: Final[dict[str, str]] = {
    "Discovery": "#60a5fa",
    "PoC": "#a78bfa",
    "Pilot": "#fbbf24",
    "Production": "#2dd4bf",
    "Scale": "#4ade80",
}


# ---------------------------------------------------------------------------
# Datenklassen
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UseCaseScore:
    """Aggregierter Score eines AI-Use-Cases.

    Attributes:
        impact: Business-Impact auf Skala 0-10.
        feasibility: Machbarkeit auf Skala 0-10.
        risk: Original-Risiko auf Skala 0-10 (10 = sehr hoch).
        risk_inverse: Invertierter Risiko-Score (10 - risk).
        score: Gewichteter Gesamtscore auf 0-100 normalisiert.
        category: Ampel-Einordnung (green/orange/red).
    """

    impact: float
    feasibility: float
    risk: float
    risk_inverse: float
    score: float
    category: TrafficLight


@dataclass(frozen=True)
class ComplianceResult:
    """Resultat der EU AI Act Compliance-Prüfung.

    Attributes:
        risk_level: Klassifizierung gemäß EU AI Act.
        dpia_required: Ob eine Datenschutz-Folgenabschätzung empfohlen wird.
        human_in_the_loop_required: Ob menschliche Aufsicht zwingend ist.
        obligations: Ausgelöste Pflichten und Empfehlungen.
    """

    risk_level: RiskLevel
    dpia_required: bool
    human_in_the_loop_required: bool
    obligations: list[str]


@dataclass(frozen=True)
class ROIResult:
    """Aggregierte ROI-Kennzahlen.

    Attributes:
        monthly_benefit: Bruttonutzen pro Monat.
        monthly_net: Nettoeffekt pro Monat (Bruttonutzen - Betriebskosten).
        total_savings: Aufsummierte Einsparungen über den Horizont.
        total_costs: Aufsummierte Kosten (Invest + Betrieb × Horizont).
        net_benefit: Differenz aus Einsparungen und Kosten.
        roi_percent: ROI in Prozent.
        break_even_months: Monat des Break-Even oder None.
    """

    monthly_benefit: float
    monthly_net: float
    total_savings: float
    total_costs: float
    net_benefit: float
    roi_percent: float
    break_even_months: float | None


# ---------------------------------------------------------------------------
# Kernlogik
# ---------------------------------------------------------------------------


def compute_use_case_score(
    impact: float,
    feasibility: float,
    risk: float,
) -> UseCaseScore:
    """Berechnet den gewichteten Use-Case-Score auf 0-100.

    Args:
        impact: Business-Impact auf Skala 0-10.
        feasibility: Machbarkeit auf Skala 0-10.
        risk: Risiko auf Skala 0-10 (10 = sehr hoch).

    Returns:
        UseCaseScore mit normalisiertem Score und Ampel-Kategorie.
    """
    risk_inverse = 10.0 - risk
    weighted = (
        impact * WEIGHT_IMPACT
        + feasibility * WEIGHT_FEASIBILITY
        + risk_inverse * WEIGHT_RISK
    ) * 10.0

    if weighted >= SCORE_GREEN:
        category: TrafficLight = "green"
    elif weighted >= SCORE_ORANGE:
        category = "orange"
    else:
        category = "red"

    return UseCaseScore(
        impact=impact,
        feasibility=feasibility,
        risk=risk,
        risk_inverse=risk_inverse,
        score=round(weighted, 1),
        category=category,
    )


def score_status(category: TrafficLight) -> tuple[str, str, str]:
    """Mappt eine Ampel-Kategorie auf Label, Themen-Farbe und Beschreibung.

    Args:
        category: Ampel-Einordnung des Scores.

    Returns:
        Tupel aus (Statuslabel, Hex-Farbe, Detailtext).
    """
    mapping: dict[TrafficLight, tuple[str, str, str]] = {
        "green": (
            "Skalierungsbereit",
            THEME.success,
            "Hohe strategische Attraktivität – Umsetzung priorisieren.",
        ),
        "orange": (
            "Pilotfähig",
            THEME.warning,
            "Solide Basis mit gezielten Klärungspunkten – Pilot empfohlen.",
        ),
        "red": (
            "Zurückstellen",
            THEME.danger,
            "Vor Umsetzung erst Nutzen und Risiken schärfen.",
        ),
    }
    return mapping[category]


def classify_eu_ai_act(
    biometric_identification: bool,
    critical_infrastructure: bool,
    education_or_employment: bool,
    law_enforcement: bool,
    essential_services: bool,
    social_scoring: bool,
    manipulation_or_exploitation: bool,
    processes_personal_data: bool,
    automated_decision: bool,
) -> ComplianceResult:
    """Klassifiziert einen Use-Case nach EU AI Act Risikopyramide.

    Args:
        biometric_identification: Echtzeit-Biometrie im öffentlichen Raum.
        critical_infrastructure: Steuerung kritischer Infrastruktur.
        education_or_employment: Bewertung im Bildungs- oder HR-Kontext.
        law_enforcement: Einsatz durch Strafverfolgungsbehörden.
        essential_services: Zugang zu Krediten, Sozialleistungen etc.
        social_scoring: Allgemeines Social Scoring durch Behörden.
        manipulation_or_exploitation: Unterschwellige Manipulation oder
            Ausnutzung schutzbedürftiger Gruppen.
        processes_personal_data: Verarbeitet personenbezogene Daten.
        automated_decision: Trifft Entscheidungen ohne menschliche Prüfung.

    Returns:
        ComplianceResult mit Risikoklasse, DPIA- und HITL-Flag sowie
        ausgelösten Pflichten.
    """
    if social_scoring or manipulation_or_exploitation:
        return ComplianceResult(
            risk_level="unacceptable",
            dpia_required=True,
            human_in_the_loop_required=True,
            obligations=[
                "Verboten gemäß Art. 5 EU AI Act – Use-Case darf nicht umgesetzt werden.",
            ],
        )

    high_risk_flags = [
        biometric_identification,
        critical_infrastructure,
        education_or_employment,
        law_enforcement,
        essential_services,
    ]
    obligations: list[str] = []

    if any(high_risk_flags):
        risk_level: RiskLevel = "high"
        human_loop = True
        obligations.extend(
            [
                "Konformitätsbewertung gemäß Art. 43 erforderlich.",
                "Eintrag in EU-Datenbank für Hochrisiko-Systeme (Art. 60).",
                "Risikomanagementsystem nach Art. 9 etablieren.",
                "Technische Dokumentation gemäß Anhang IV bereitstellen.",
            ]
        )
    else:
        risk_level = "limited"
        human_loop = automated_decision

    dpia = processes_personal_data and (risk_level == "high" or automated_decision)
    if dpia:
        obligations.append(
            "DPIA (Datenschutz-Folgenabschätzung) gemäß Art. 35 DSGVO durchführen."
        )
    if human_loop:
        obligations.append(
            "Human-in-the-Loop verpflichtend – qualifizierte menschliche Aufsicht."
        )
    if risk_level == "limited":
        obligations.append(
            "Transparenzpflicht (Art. 52): Nutzer:innen über AI-Interaktion informieren."
        )

    return ComplianceResult(
        risk_level=risk_level,
        dpia_required=dpia,
        human_in_the_loop_required=human_loop,
        obligations=obligations,
    )


def compliance_status(level: RiskLevel) -> tuple[str, str, str]:
    """Mappt eine Risikoklasse auf Label, Farbe und Beschreibung.

    Args:
        level: EU AI Act Risikoklasse.

    Returns:
        Tupel aus (Statuslabel, Hex-Farbe, Detailtext).
    """
    mapping: dict[RiskLevel, tuple[str, str, str]] = {
        "unacceptable": (
            "Unzulässig (Art. 5)",
            THEME.critical,
            "Verbotene Praxis – Umsetzung ist rechtlich ausgeschlossen.",
        ),
        "high": (
            "Hochrisiko (Anhang III)",
            THEME.danger,
            "Vor Produktivbetrieb braucht es Konformitätsbewertung und HITL.",
        ),
        "limited": (
            "Begrenztes Risiko",
            THEME.warning,
            "Machbar mit Transparenzpflicht und klaren Kontrollen.",
        ),
        "minimal": (
            "Minimales Risiko",
            THEME.success,
            "Geringe regulatorische Last – Standard-Governance ausreichend.",
        ),
    }
    return mapping[level]


def compute_roi(
    initial_investment: float,
    monthly_operating_cost: float,
    monthly_hours_saved: float,
    hourly_rate: float,
    monthly_revenue_uplift: float,
    monthly_risk_reduction: float,
    horizon_months: int,
) -> ROIResult:
    """Berechnet ROI, Net Benefit und Break-Even-Punkt.

    Args:
        initial_investment: Einmalige Investition in Euro.
        monthly_operating_cost: Laufende Betriebskosten pro Monat.
        monthly_hours_saved: Eingesparte Stunden pro Monat.
        hourly_rate: Interner Stundensatz in Euro.
        monthly_revenue_uplift: Zusätzlicher Umsatz/Marge pro Monat.
        monthly_risk_reduction: Vermiedene Risiko-/Fehlerkosten pro Monat.
        horizon_months: Betrachtungszeitraum in Monaten (>= 1).

    Returns:
        ROIResult mit aggregierten Kennzahlen. ``break_even_months`` ist
        ``None``, wenn der Nettoeffekt pro Monat nicht positiv ist.
    """
    horizon = max(1, horizon_months)
    monthly_benefit = (
        monthly_hours_saved * hourly_rate
        + monthly_revenue_uplift
        + monthly_risk_reduction
    )
    monthly_net = monthly_benefit - monthly_operating_cost
    total_costs = initial_investment + monthly_operating_cost * horizon
    total_savings = monthly_benefit * horizon
    net_benefit = total_savings - total_costs
    roi_percent = (net_benefit / total_costs * 100.0) if total_costs > 0 else 0.0
    break_even = (initial_investment / monthly_net) if monthly_net > 0 else None

    return ROIResult(
        monthly_benefit=monthly_benefit,
        monthly_net=monthly_net,
        total_savings=total_savings,
        total_costs=total_costs,
        net_benefit=net_benefit,
        roi_percent=round(roi_percent, 1),
        break_even_months=None if break_even is None else round(break_even, 1),
    )


def build_roadmap(
    start: date,
    complexity: float,
    risk: float,
) -> pd.DataFrame:
    """Erzeugt eine fünfphasige Rollout-Roadmap mit dynamischer Dauer.

    Multiplikator pro Phase::

        multiplier = (1 + complexity/10) × (1 + 0.5 × risk/10)

    Args:
        start: Startdatum der Discovery-Phase.
        complexity: Komplexität auf Skala 0-10.
        risk: Risiko auf Skala 0-10.

    Returns:
        DataFrame mit ``Phase``, ``Owner``, ``Start``, ``Finish``,
        ``Dauer (Tage)``.
    """
    multiplier = (1.0 + complexity / 10.0) * (1.0 + 0.5 * risk / 10.0)
    rows: list[dict[str, object]] = []
    current = pd.Timestamp(start)
    for phase, base_days in PHASE_BASE_DAYS.items():
        duration = max(1, int(round(base_days * multiplier)))
        finish = current + pd.Timedelta(days=duration)
        rows.append(
            {
                "Phase": phase,
                "Owner": PHASE_OWNERS[phase],
                "Start": current,
                "Finish": finish,
                "Dauer (Tage)": duration,
            }
        )
        current = finish
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# UI Setup
# ---------------------------------------------------------------------------


st.set_page_config(
    page_title="AI Strategy Toolkit",
    page_icon=":material/psychology:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_theme() -> None:
    """Injektiert globales Stylesheet mit Design-Tokens als CSS-Variablen."""
    st.markdown(
        f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

            :root {{
                --bg: {THEME.bg};
                --surface: {THEME.surface};
                --surface-elev: {THEME.surface_elevated};
                --border: {THEME.border};
                --border-strong: {THEME.border_strong};
                --text: {THEME.text_primary};
                --text-2: {THEME.text_secondary};
                --text-3: {THEME.text_muted};
                --accent: {THEME.accent};
                --accent-soft: {THEME.accent_soft};
                --success: {THEME.success};
                --warning: {THEME.warning};
                --danger: {THEME.danger};
                --critical: {THEME.critical};
                --radius: 12px;
                --radius-sm: 8px;
                --shadow: 0 1px 2px rgba(0,0,0,0.20), 0 8px 24px rgba(0,0,0,0.18);
            }}

            html, body, [class*="css"], .stApp {{
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
                font-feature-settings: 'cv11', 'ss01', 'ss03';
            }}

            .stApp {{
                background:
                    radial-gradient(900px 480px at 12% -10%, rgba(45, 212, 191, 0.08), transparent 60%),
                    radial-gradient(700px 420px at 100% 0%, rgba(96, 165, 250, 0.06), transparent 55%),
                    var(--bg);
                color: var(--text);
            }}

            /* Sidebar */
            section[data-testid="stSidebar"] {{
                background: var(--surface);
                border-right: 1px solid var(--border);
            }}
            section[data-testid="stSidebar"] * {{ color: var(--text); }}
            section[data-testid="stSidebar"] [role="radiogroup"] label {{
                border-radius: var(--radius-sm);
                padding: 0.45rem 0.6rem;
                margin: 0.1rem 0;
                transition: background 120ms ease;
            }}
            section[data-testid="stSidebar"] [role="radiogroup"] label:hover {{
                background: var(--accent-soft);
            }}

            .brand {{
                display: flex;
                align-items: center;
                gap: 0.7rem;
                padding: 0.4rem 0 1.25rem;
                border-bottom: 1px solid var(--border);
                margin-bottom: 1.2rem;
            }}
            .brand-mark {{
                width: 36px; height: 36px;
                border-radius: 9px;
                background: linear-gradient(135deg, var(--accent), #60a5fa);
                display: grid; place-items: center;
                color: #0b1120;
                font-weight: 700;
                font-size: 0.95rem;
                letter-spacing: -0.02em;
                box-shadow: 0 6px 18px rgba(45, 212, 191, 0.25);
            }}
            .brand-text {{ display: flex; flex-direction: column; line-height: 1.15; }}
            .brand-text .name {{ font-weight: 600; font-size: 0.95rem; color: var(--text); }}
            .brand-text .tag {{ font-size: 0.72rem; color: var(--text-3); letter-spacing: 0.04em; }}

            .nav-label {{
                font-size: 0.7rem;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: var(--text-3);
                margin: 0.4rem 0 0.4rem;
            }}

            .sidebar-foot {{
                margin-top: 1.4rem;
                padding-top: 1rem;
                border-top: 1px solid var(--border);
                color: var(--text-3);
                font-size: 0.78rem;
                line-height: 1.55;
            }}

            /* Hero */
            .hero {{
                padding: 1.25rem 0 1.6rem;
                margin-bottom: 1.4rem;
                border-bottom: 1px solid var(--border);
            }}
            .hero .eyebrow {{
                display: inline-block;
                color: var(--accent);
                background: var(--accent-soft);
                padding: 0.28rem 0.7rem;
                border-radius: 999px;
                font-size: 0.7rem;
                font-weight: 600;
                letter-spacing: 0.14em;
                text-transform: uppercase;
                margin-bottom: 0.95rem;
            }}
            .hero h1 {{
                margin: 0 0 0.55rem;
                font-size: clamp(1.85rem, 3vw, 2.45rem);
                font-weight: 600;
                letter-spacing: -0.025em;
                line-height: 1.1;
                color: var(--text);
            }}
            .hero p {{
                margin: 0;
                max-width: 740px;
                color: var(--text-2);
                font-size: 1.02rem;
                line-height: 1.55;
            }}

            /* Section title */
            h2, h3, .stSubheader {{
                color: var(--text);
                font-weight: 600;
                letter-spacing: -0.01em;
            }}
            h3, .stSubheader {{ font-size: 1.05rem !important; }}

            /* Metric cards */
            div[data-testid="stMetric"] {{
                background: var(--surface-elev);
                border: 1px solid var(--border);
                border-radius: var(--radius);
                padding: 1.05rem 1.15rem;
                box-shadow: var(--shadow);
                transition: border-color 120ms ease;
            }}
            div[data-testid="stMetric"]:hover {{ border-color: var(--border-strong); }}
            div[data-testid="stMetricLabel"] {{
                color: var(--text-2);
                font-size: 0.78rem !important;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.06em;
            }}
            div[data-testid="stMetricValue"] {{
                color: var(--text);
                font-weight: 600;
                letter-spacing: -0.02em;
            }}
            div[data-testid="stMetricDelta"] {{ font-weight: 500; }}

            /* Status ribbon */
            .status-ribbon {{
                background: var(--surface-elev);
                border: 1px solid var(--border);
                border-left: 4px solid var(--status-color);
                border-radius: var(--radius);
                padding: 1rem 1.2rem;
                box-shadow: var(--shadow);
            }}
            .status-ribbon .label {{
                color: var(--status-color);
                font-weight: 600;
                font-size: 0.95rem;
                letter-spacing: -0.005em;
            }}
            .status-ribbon .detail {{
                color: var(--text-2);
                font-size: 0.88rem;
                margin-top: 0.3rem;
                line-height: 1.5;
            }}

            /* Obligation chip */
            .obl-list {{ display: flex; flex-direction: column; gap: 0.5rem; margin-top: 0.4rem; }}
            .obl {{
                display: flex; align-items: flex-start; gap: 0.65rem;
                background: var(--surface-elev);
                border: 1px solid var(--border);
                border-left: 3px solid var(--obl-color);
                border-radius: var(--radius-sm);
                padding: 0.7rem 0.85rem;
                color: var(--text);
                font-size: 0.9rem;
                line-height: 1.5;
            }}
            .obl::before {{
                content: "";
                display: block;
                width: 6px; height: 6px;
                border-radius: 999px;
                background: var(--obl-color);
                margin-top: 0.5rem;
                flex-shrink: 0;
            }}

            /* Inputs */
            div[data-baseweb="input"] input,
            div[data-baseweb="select"] {{ font-family: 'Inter', sans-serif; }}
            .stSlider [data-baseweb="slider"] {{ padding-top: 0.3rem; }}

            /* Buttons */
            .stButton > button, .stDownloadButton > button {{
                border-radius: var(--radius-sm);
                font-weight: 500;
                letter-spacing: 0.005em;
                border: 1px solid var(--border-strong);
                transition: transform 80ms ease, border-color 120ms ease;
            }}
            .stButton > button:hover, .stDownloadButton > button:hover {{
                border-color: var(--accent);
                transform: translateY(-1px);
            }}

            /* Dataframe */
            div[data-testid="stDataFrame"] {{
                border: 1px solid var(--border);
                border-radius: var(--radius);
                overflow: hidden;
            }}

            /* Expander */
            details[data-testid="stExpander"] {{
                background: var(--surface-elev);
                border: 1px solid var(--border);
                border-radius: var(--radius);
            }}

            /* Code blocks inside markdown */
            code, pre {{
                font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace !important;
                font-size: 0.85rem;
            }}

            /* Hide Streamlit header chrome for cleaner look */
            header[data-testid="stHeader"] {{ background: transparent; }}

            /* Divider */
            hr {{ border-color: var(--border) !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_theme()


if "saved_scores" not in st.session_state:
    st.session_state.saved_scores = []


# ---------------------------------------------------------------------------
# UI Helper
# ---------------------------------------------------------------------------


def render_brand() -> None:
    """Rendert das Brand-Lockup oben in der Sidebar."""
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">AI</div>
            <div class="brand-text">
                <span class="name">Strategy Toolkit</span>
                <span class="tag">DECISION INTELLIGENCE</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header(eyebrow: str, title: str, subtitle: str) -> None:
    """Rendert den Hero-Header eines Moduls.

    Args:
        eyebrow: Kurzes Label über dem Titel (Modulkennung).
        title: Haupttitel.
        subtitle: Beschreibungstext unter dem Titel.
    """
    st.markdown(
        f"""
        <div class="hero">
            <span class="eyebrow">{eyebrow}</span>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status(label: str, color: str, detail: str) -> None:
    """Rendert ein farbiges Status-Ribbon.

    Args:
        label: Statuslabel (in Akzentfarbe).
        color: Hex-Farbe für Border und Label.
        detail: Beschreibungstext.
    """
    st.markdown(
        f"""
        <div class="status-ribbon" style="--status-color: {color};">
            <div class="label">{label}</div>
            <div class="detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_obligations(items: list[str], color: str) -> None:
    """Rendert eine Liste regulatorischer Pflichten als Chips.

    Args:
        items: Liste von Pflicht-/Empfehlungstexten.
        color: Hex-Farbe für linke Border und Bullet.
    """
    if not items:
        return
    rows = "".join(f'<div class="obl">{item}</div>' for item in items)
    st.markdown(
        f'<div class="obl-list" style="--obl-color: {color};">{rows}</div>',
        unsafe_allow_html=True,
    )


def themed_layout(fig: go.Figure, title: str | None = None) -> go.Figure:
    """Wendet das dunkle Theme auf eine Plotly-Figur an.

    Args:
        fig: Plotly-Figur, die mutiert und zurückgegeben wird.
        title: Optionaler Titel. Wenn None, wird kein Titel gesetzt.

    Returns:
        Dieselbe Figur mit angewendetem Theme.
    """
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            family="Inter, system-ui, sans-serif",
            color=THEME.text_primary,
            size=12,
        ),
        title=dict(
            text=title or "",
            font=dict(size=14, color=THEME.text_primary),
            x=0.0,
            xanchor="left",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=THEME.border,
            borderwidth=0,
            font=dict(color=THEME.text_secondary),
        ),
        hoverlabel=dict(
            bgcolor=THEME.surface_elevated,
            bordercolor=THEME.border_strong,
            font=dict(color=THEME.text_primary, family="Inter, sans-serif"),
        ),
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        colorway=[
            THEME.accent, "#60a5fa", "#a78bfa", "#fbbf24", "#f87171",
        ],
    )
    fig.update_xaxes(
        gridcolor=THEME.border,
        linecolor=THEME.border_strong,
        zerolinecolor=THEME.border_strong,
        title_font=dict(color=THEME.text_secondary, size=11),
        tickfont=dict(color=THEME.text_secondary),
    )
    fig.update_yaxes(
        gridcolor=THEME.border,
        linecolor=THEME.border_strong,
        zerolinecolor=THEME.border_strong,
        title_font=dict(color=THEME.text_secondary, size=11),
        tickfont=dict(color=THEME.text_secondary),
    )
    return fig


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


def scorer_view() -> None:
    """Rendert das Use-Case Scoring Modul (40/30/30, Skala 0-100)."""
    render_header(
        eyebrow="Modul 01 · Prioritization",
        title="AI Opportunity Scorer",
        subtitle=(
            "Priorisiere Use Cases nach Impact (40 %), Feasibility (30 %) und "
            "Risiko (30 %). Score skaliert auf 0–100."
        ),
    )

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        impact = st.slider(
            "Business Impact",
            min_value=0.0, max_value=10.0, value=7.0, step=0.5,
            help="Erwarteter Beitrag zu Umsatz, Kosten, Qualität oder Geschwindigkeit.",
        )
        feasibility = st.slider(
            "Feasibility",
            min_value=0.0, max_value=10.0, value=6.0, step=0.5,
            help="Datenreife, technische Machbarkeit, Integrationsaufwand, Team-Readiness.",
        )
        risk = st.slider(
            "Risiko",
            min_value=0.0, max_value=10.0, value=4.0, step=0.5,
            help="Regulatorisch, ethisch, reputationsbezogen, technisch.",
        )

    result = compute_use_case_score(impact, feasibility, risk)
    label, color, detail = score_status(result.category)
    previous = (
        st.session_state.saved_scores[-1]["Score"]
        if st.session_state.saved_scores
        else None
    )
    delta = None if previous is None else round(result.score - previous, 1)

    with right:
        st.metric(
            "Composite Score",
            f"{result.score:.1f} / 100",
            delta=None if delta is None else f"{delta:+.1f} vs. letzter Save",
            delta_color="normal",
        )
        render_status(label, color, detail)
        if st.button("Score sichern", type="primary", use_container_width=True):
            st.session_state.saved_scores.append(
                {
                    "Impact": result.impact,
                    "Feasibility": result.feasibility,
                    "Risiko": result.risk,
                    "Score": result.score,
                    "Status": label,
                }
            )
            st.toast(f"Score {result.score:.1f}/100 gesichert", icon="✓")

    st.subheader("Score-Aufschlüsselung")
    breakdown = pd.DataFrame(
        {
            "Dimension": ["Impact", "Feasibility", "Risiko (invertiert)"],
            "Wert (0-10)": [result.impact, result.feasibility, result.risk_inverse],
            "Gewicht": [WEIGHT_IMPACT, WEIGHT_FEASIBILITY, WEIGHT_RISK],
            "Beitrag (0-100)": [
                round(result.impact * WEIGHT_IMPACT * 10, 1),
                round(result.feasibility * WEIGHT_FEASIBILITY * 10, 1),
                round(result.risk_inverse * WEIGHT_RISK * 10, 1),
            ],
        }
    )

    bar = px.bar(
        breakdown,
        x="Beitrag (0-100)",
        y="Dimension",
        orientation="h",
        text="Beitrag (0-100)",
        color="Dimension",
        color_discrete_sequence=[THEME.accent, "#60a5fa", "#a78bfa"],
    )
    bar.update_traces(textposition="outside", cliponaxis=False)
    bar.update_layout(showlegend=False, height=240)
    themed_layout(bar)
    bar.update_yaxes(autorange="reversed")
    st.plotly_chart(bar, use_container_width=True)

    st.dataframe(breakdown, use_container_width=True, hide_index=True)

    if st.session_state.saved_scores:
        st.subheader("Gesicherte Scores")
        st.dataframe(
            pd.DataFrame(st.session_state.saved_scores),
            use_container_width=True,
            hide_index=True,
        )


def compliance_view() -> None:
    """Rendert die EU AI Act Compliance-Prüfung mit Checkboxen."""
    render_header(
        eyebrow="Modul 02 · Regulation",
        title="EU AI Act Compliance Check",
        subtitle=(
            "Strukturierte Vorprüfung inklusive DPIA-Empfehlung und "
            "Human-in-the-Loop-Indikation. Ersetzt keine juristische Beratung."
        ),
    )

    st.subheader("Verbotene Praktiken (Art. 5)")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        social_scoring = st.checkbox(
            "Social Scoring durch Behörden",
            help="Allgemeine Bewertung von Vertrauenswürdigkeit über mehrere Lebensbereiche.",
        )
    with col_p2:
        manipulation = st.checkbox(
            "Unterschwellige Manipulation oder Ausnutzung Schutzbedürftiger",
        )

    st.subheader("Hochrisiko-Trigger (Anhang III)")
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        biometric = st.checkbox("Biometrische Identifikation im öffentlichen Raum")
        critical = st.checkbox("Steuerung kritischer Infrastruktur")
        education = st.checkbox("Bildung, Prüfungen oder Personalauswahl")
    with col_h2:
        law_enf = st.checkbox("Einsatz durch Strafverfolgungsbehörden")
        essential = st.checkbox(
            "Zugang zu essenziellen Diensten (Kredit, Sozialleistung)"
        )

    st.subheader("Datenschutz & Entscheidungslogik")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        personal = st.checkbox("Verarbeitet personenbezogene Daten", value=True)
    with col_d2:
        automated = st.checkbox(
            "Trifft automatisierte Entscheidungen ohne menschliche Prüfung"
        )

    result = classify_eu_ai_act(
        biometric_identification=biometric,
        critical_infrastructure=critical,
        education_or_employment=education,
        law_enforcement=law_enf,
        essential_services=essential,
        social_scoring=social_scoring,
        manipulation_or_exploitation=manipulation,
        processes_personal_data=personal,
        automated_decision=automated,
    )
    label, color, detail = compliance_status(result.risk_level)

    st.divider()
    metric_col, text_col = st.columns([0.95, 1.05], gap="large")
    with metric_col:
        m1, m2 = st.columns(2)
        m1.metric("DPIA empfohlen", "Ja" if result.dpia_required else "Nein")
        m2.metric(
            "Human-in-the-Loop",
            "Pflicht" if result.human_in_the_loop_required else "Optional",
        )
        render_status(f"Risikoklasse: {label}", color, detail)

    with text_col:
        st.subheader("Pflichten & Empfehlungen")
        if not result.obligations:
            render_status(
                "Keine spezifischen Pflichten",
                THEME.success,
                "Minimales Risikoprofil. Dokumentiere Zweck, Datenquellen, "
                "Verantwortliche und Monitoring trotzdem sauber.",
            )
        else:
            render_obligations(result.obligations, color)


def roi_view() -> None:
    """Rendert den ROI- und Break-Even-Rechner inkl. CSV-Export."""
    render_header(
        eyebrow="Modul 03 · Economics",
        title="ROI & Break-Even Rechner",
        subtitle=(
            "Simuliere den wirtschaftlichen Effekt eines AI Use Cases mit "
            "sofort aktualisierten Kennzahlen und CSV-Export."
        ),
    )

    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        initial_investment = st.number_input(
            "Einmalige Investition (EUR)",
            min_value=0.0, value=65_000.0, step=5_000.0, format="%.2f",
        )
        monthly_operating_cost = st.number_input(
            "Monatliche Betriebskosten (EUR)",
            min_value=0.0, value=6_500.0, step=500.0, format="%.2f",
        )
        monthly_hours_saved = st.number_input(
            "Eingesparte Stunden pro Monat",
            min_value=0.0, value=420.0, step=10.0, format="%.1f",
        )
        horizon = st.number_input(
            "Betrachtungszeitraum (Monate)",
            min_value=1, max_value=120, value=24, step=1,
        )

    with col_b:
        hourly_rate = st.number_input(
            "Interner Stundensatz (EUR)",
            min_value=0.0, value=85.0, step=5.0, format="%.2f",
        )
        monthly_revenue_uplift = st.number_input(
            "Monatlicher Umsatz-/Marge-Uplift (EUR)",
            min_value=0.0, value=12_000.0, step=1_000.0, format="%.2f",
        )
        monthly_risk_reduction = st.number_input(
            "Monatlich vermiedene Risiko-/Fehlerkosten (EUR)",
            min_value=0.0, value=4_000.0, step=500.0, format="%.2f",
        )

    result = compute_roi(
        initial_investment=initial_investment,
        monthly_operating_cost=monthly_operating_cost,
        monthly_hours_saved=monthly_hours_saved,
        hourly_rate=hourly_rate,
        monthly_revenue_uplift=monthly_revenue_uplift,
        monthly_risk_reduction=monthly_risk_reduction,
        horizon_months=int(horizon),
    )

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Net pro Monat", f"{result.monthly_net:,.0f} €")
    m2.metric("Net Benefit gesamt", f"{result.net_benefit:,.0f} €")
    m3.metric("ROI", f"{result.roi_percent:,.1f} %")
    if result.break_even_months is None:
        m4.metric("Break-Even", "—", help="Nettoeffekt pro Monat ist nicht positiv.")
    else:
        m4.metric("Break-Even", f"{result.break_even_months:,.1f} Monate")

    months = list(range(0, int(horizon) + 1))
    projection = pd.DataFrame(
        {
            "Monat": months,
            "Kumulierte Einsparung (€)": [result.monthly_benefit * m for m in months],
            "Kumulierte Kosten (€)": [
                initial_investment + monthly_operating_cost * m for m in months
            ],
        }
    )
    projection["Net Benefit (€)"] = (
        projection["Kumulierte Einsparung (€)"]
        - projection["Kumulierte Kosten (€)"]
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=projection["Monat"], y=projection["Kumulierte Einsparung (€)"],
            mode="lines", name="Einsparung",
            line=dict(color=THEME.success, width=2.5),
            hovertemplate="Monat %{x}: %{y:,.0f} €<extra>Einsparung</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=projection["Monat"], y=projection["Kumulierte Kosten (€)"],
            mode="lines", name="Kosten",
            line=dict(color=THEME.danger, width=2.5),
            hovertemplate="Monat %{x}: %{y:,.0f} €<extra>Kosten</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=projection["Monat"], y=projection["Net Benefit (€)"],
            mode="lines", name="Net Benefit",
            line=dict(color=THEME.accent, width=3),
            fill="tozeroy",
            fillcolor="rgba(45, 212, 191, 0.10)",
            hovertemplate="Monat %{x}: %{y:,.0f} €<extra>Net Benefit</extra>",
        )
    )
    if result.break_even_months is not None and result.break_even_months <= horizon:
        fig.add_vline(
            x=result.break_even_months,
            line=dict(color=THEME.warning, dash="dash", width=1.5),
            annotation_text=f"Break-Even · M{result.break_even_months:.1f}",
            annotation_position="top",
            annotation_font_color=THEME.warning,
        )
    themed_layout(fig, title="Kumulierte Entwicklung über den Horizont")
    fig.update_yaxes(tickformat=",.0f", title_text="EUR")
    fig.update_xaxes(title_text="Monat")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Berechnungsgrundlage anzeigen"):
        st.markdown(
            f"""
```
monthly_benefit = hours_saved × rate + revenue_uplift + risk_reduction
                = {monthly_hours_saved:,.1f} × {hourly_rate:,.2f} + {monthly_revenue_uplift:,.0f} + {monthly_risk_reduction:,.0f}
                = {result.monthly_benefit:,.0f} EUR

monthly_net     = monthly_benefit − monthly_operating_cost
                = {result.monthly_net:,.0f} EUR

total_costs     = initial_investment + monthly_operating_cost × horizon
                = {initial_investment:,.0f} + {monthly_operating_cost:,.0f} × {int(horizon)}
                = {result.total_costs:,.0f} EUR

total_savings   = monthly_benefit × horizon = {result.total_savings:,.0f} EUR
net_benefit     = total_savings − total_costs = {result.net_benefit:,.0f} EUR
ROI %           = net_benefit / total_costs × 100 = {result.roi_percent:.1f} %
break_even_m    = initial_investment / monthly_net
```
            """
        )

    summary = pd.DataFrame(
        [
            {"Kennzahl": "Einmalige Investition", "Wert (EUR)": initial_investment},
            {"Kennzahl": "Monatliche Betriebskosten", "Wert (EUR)": monthly_operating_cost},
            {"Kennzahl": "Monatlicher Bruttonutzen", "Wert (EUR)": result.monthly_benefit},
            {"Kennzahl": "Monatlicher Nettoeffekt", "Wert (EUR)": result.monthly_net},
            {"Kennzahl": "Gesamt Net Benefit", "Wert (EUR)": result.net_benefit},
            {"Kennzahl": "ROI (%)", "Wert (EUR)": result.roi_percent},
            {
                "Kennzahl": "Break-Even (Monate)",
                "Wert (EUR)": float("nan") if result.break_even_months is None else result.break_even_months,
            },
        ]
    )
    st.subheader("Zusammenfassung")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    csv_buffer = StringIO()
    summary.to_csv(csv_buffer, index=False)
    st.download_button(
        "CSV exportieren",
        data=csv_buffer.getvalue(),
        file_name="ai_roi_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )


def roadmap_view() -> None:
    """Rendert den Roadmap-Generator mit dynamischer Phasendauer."""
    render_header(
        eyebrow="Modul 04 · Delivery",
        title="AI Implementation Roadmap",
        subtitle=(
            "Fünf Phasen (Discovery → PoC → Pilot → Production → Scale) mit "
            "dynamischer Dauer basierend auf Komplexität und Risiko."
        ),
    )

    col_a, col_b, col_c = st.columns(3, gap="large")
    with col_a:
        start = st.date_input("Projektstart", value=date.today())
    with col_b:
        complexity = st.slider(
            "Komplexität", 0.0, 10.0, 5.0, 0.5,
            help="Architektur, Integrationen, Team-Skalierung, Stakeholder-Anzahl.",
        )
    with col_c:
        risk = st.slider(
            "Risiko", 0.0, 10.0, 4.0, 0.5,
            help="Regulatorisch, technisch, Change-Management.",
        )

    roadmap = build_roadmap(start, complexity, risk)
    total_days = int(roadmap["Dauer (Tage)"].sum())
    end_date = roadmap["Finish"].max().date()

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Gesamtdauer", f"{total_days} Tage")
    m2.metric("Geschätzte Monate", f"~ {total_days // 30}")
    m3.metric("Voraussichtliches Ende", end_date.isoformat())

    fig = px.timeline(
        roadmap,
        x_start="Start",
        x_end="Finish",
        y="Phase",
        color="Phase",
        hover_data={"Owner": True, "Dauer (Tage)": True, "Start": False, "Finish": False},
        color_discrete_map=PHASE_COLORS,
    )
    fig.update_yaxes(autorange="reversed", title_text="")
    fig.update_xaxes(title_text="")
    fig.update_traces(marker_line_width=0)
    themed_layout(fig, title="Rollout-Timeline")
    fig.update_layout(height=360, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Phasenübersicht")
    display = roadmap.copy()
    display["Start"] = display["Start"].dt.strftime("%Y-%m-%d")
    display["Finish"] = display["Finish"].dt.strftime("%Y-%m-%d")
    st.dataframe(display, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Sidebar & Dispatch
# ---------------------------------------------------------------------------


with st.sidebar:
    render_brand()
    st.markdown('<div class="nav-label">Module</div>', unsafe_allow_html=True)
    view = st.radio(
        "Modul",
        ["Use-Case Scorer", "Risk & Compliance", "ROI Calculator", "Roadmap Generator"],
        label_visibility="collapsed",
    )
    st.markdown(
        """
        <div class="sidebar-foot">
            <strong style="color: #cbd5e1;">Methodik</strong><br>
            Scorer · 40 / 30 / 30 (Impact / Feasibility / Risk)<br>
            Compliance · EU AI Act + DSGVO Art. 35<br>
            ROI · lineare Kostenprojektion mit Break-Even<br>
            Roadmap · 5 Phasen, dynamisch skaliert
        </div>
        """,
        unsafe_allow_html=True,
    )


if view == "Use-Case Scorer":
    scorer_view()
elif view == "Risk & Compliance":
    compliance_view()
elif view == "ROI Calculator":
    roi_view()
else:
    roadmap_view()
