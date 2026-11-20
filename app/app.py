"""
AgriVision AI - Plant Nutrition Deficiency Detection System
Full Web Application:
- Public Experience: Professional SaaS Landing Page & Secure Authentication (Login / Register / Password Recovery)
- Protected Experience: Comprehensive Diagnostic Workstation, Grad-CAM Explainable AI, Research Evaluation, & Encyclopedia
- Preserves 100% of existing functionality, models, metrics, bilingual support, and Nanstack branding.
"""

import sys
from pathlib import Path
import json
import base64
import streamlit as st
from PIL import Image
import numpy as np

# Ensure project root is in python path
FILE_PATH = Path(__file__).resolve()
ROOT_DIR = FILE_PATH.parent if (FILE_PATH.parent / "src").exists() else FILE_PATH.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Institution & Branding Constants
UON_LOGO_PATH = ROOT_DIR / "app" / "assets" / "uon_logo.png"
NANSTACK_LOGO_PATH = ROOT_DIR / "app" / "assets" / "nanstack_logo.jpg"

def get_base64_image(image_path: Path) -> str:
    """Reads and encodes an image file to a base64 string for direct HTML embedding."""
    if image_path.exists():
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""

from src.config import (
    SAVED_MODEL_PATH,
    HISTORY_PATH,
    METRICS_PATH,
    CONFUSION_MATRIX_PATH,
    CONFIDENCE_THRESHOLD,
    TARGET_CLASSES
)
from src.inference import NutrientDeficiencyPredictor
from src.knowledge_base import NUTRIENT_KNOWLEDGE_BASE, DISCLAIMER_TEXT, DISCLAIMER_TEXT_UR
from src.auth import authenticate_user, register_user, reset_password

# Professional SVG Vector Icon Set (Lucide Icons)
ICONS = {
    "leaf": '<svg class="icon" viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/></svg>',
    "check_circle": '<svg class="icon" viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>',
    "alert_triangle": '<svg class="icon" viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" x2="12" y1="9" y2="13"/><line x1="12" x2="12.01" y1="17" y2="17"/></svg>',
    "microscope": '<svg class="icon" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 18h8"/><path d="M3 22h18"/><path d="m14 22 .75-3.01"/><path d="M10 22v-3"/><path d="M9 13.5 15.5 7"/><path d="m17 4.5 1.5 1.5"/><path d="m13 1.5 6 6"/><path d="m10 9-2.5 2.5"/><path d="m7.5 11.5-3 3a2.12 2.12 0 0 0 3 3l3-3"/></svg>',
    "scan": '<svg class="icon" viewBox="0 0 24 24" width="19" height="19" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><circle cx="12" cy="12" r="3"/><path d="m16 16-.9-.9"/></svg>',
    "clipboard": '<svg class="icon" viewBox="0 0 24 24" width="19" height="19" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="8" height="4" x="8" y="2" rx="1" ry="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><path d="m9 14 2 2 4-4"/></svg>',
    "bar_chart": '<svg class="icon" viewBox="0 0 24 24" width="19" height="19" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" x2="12" y1="20" y2="10"/><line x1="18" x2="18" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="16"/></svg>',
    "book": '<svg class="icon" viewBox="0 0 24 24" width="19" height="19" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>',
    "globe": '<svg class="icon" viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" x2="22" y1="12" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>',
    "layers": '<svg class="icon" viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
    "cpu": '<svg class="icon" viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/></svg>',
    "upload": '<svg class="icon" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/></svg>',
    "upload_large": '<svg class="icon" viewBox="0 0 24 24" width="46" height="46" fill="none" stroke="#94a3b8" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/></svg>',
    "shield": '<svg class="icon" viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
    "target": '<svg class="icon" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    "flask": '<svg class="icon" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 2v7.527a2 2 0 0 1-.211.896L4.72 20.55a1 1 0 0 0 .9 1.45h12.76a1 1 0 0 0 .9-1.45l-5.069-10.127A2 2 0 0 1 14 9.527V2"/><path d="M8.5 2h7"/><path d="M7 16h10"/></svg>',
    "activity": '<svg class="icon" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    "lock": '<svg class="icon" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>',
    "user": '<svg class="icon" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
    "log_out": '<svg class="icon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" x2="9" y1="12" y2="12"/></svg>',
    "check": '<svg class="icon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>'
}

# Set page configuration
st.set_page_config(
    page_title="AI Based Plant Nutrients Deficiency Detection",
    page_icon=str(UON_LOGO_PATH) if UON_LOGO_PATH.exists() else "🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Web App CSS for sleek SaaS look and mobile responsiveness
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Noto+Nastaliq+Urdu:wght@400;700&display=swap');

    /* Global Typography & Layout */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide Streamlit default branding, decoration, and right-side toolbar items (Deploy, 3-dots menu) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stDecoration"] {display: none !important;}
    .stAppDeployButton,
    [data-testid="stToolbarActions"],
    [data-testid="stToolbarActionButton"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* Keep toolbar and header transparent and non-blocking except for interactive buttons */
    [data-testid="stToolbar"] {
        background: transparent !important;
        pointer-events: none !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
        pointer-events: none !important;
        z-index: 99999 !important;
    }
    header[data-testid="stHeader"] button,
    header[data-testid="stHeader"] [data-testid="stExpandSidebarButton"],
    header[data-testid="stHeader"] [data-testid="stSidebarCollapsedControl"] {
        pointer-events: auto !important;
    }

    /* Style the native Streamlit Sidebar Open button at Top-Left Corner */
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"] {
        position: fixed !important;
        top: 14px !important;
        left: 14px !important;
        z-index: 9999999 !important;
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
    }

    [data-testid="stExpandSidebarButton"] button,
    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="collapsedControl"] button {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 42px !important;
        height: 42px !important;
        background: #ffffff !important;
        color: #065f46 !important;
        border: 1.5px solid #a7f3d0 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08), 0 1px 3px rgba(0, 0, 0, 0.04) !important;
        cursor: pointer !important;
        transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1) !important;
        outline: none !important;
        pointer-events: auto !important;
        visibility: visible !important;
    }

    [data-testid="stExpandSidebarButton"] button:hover,
    [data-testid="stSidebarCollapsedControl"] button:hover,
    [data-testid="collapsedControl"] button:hover {
        background: #ecfdf5 !important;
        color: #047857 !important;
        border-color: #34d399 !important;
        box-shadow: 0 6px 20px rgba(5, 150, 105, 0.22) !important;
        transform: scale(1.06) !important;
    }

    /* Replace default chevron arrow with pure CSS 3-bar hamburger menu icon */
    [data-testid="stExpandSidebarButton"] button svg,
    [data-testid="stSidebarCollapsedControl"] button svg,
    [data-testid="collapsedControl"] button svg {
        display: none !important;
    }
    [data-testid="stExpandSidebarButton"] button::before,
    [data-testid="stSidebarCollapsedControl"] button::before,
    [data-testid="collapsedControl"] button::before {
        content: "" !important;
        display: block !important;
        width: 18px !important;
        height: 2.2px !important;
        background-color: #065f46 !important;
        border-radius: 2px !important;
        box-shadow: 0 -6px 0 #065f46, 0 6px 0 #065f46 !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stExpandSidebarButton"] button:hover::before,
    [data-testid="stSidebarCollapsedControl"] button:hover::before,
    [data-testid="collapsedControl"] button:hover::before {
        background-color: #047857 !important;
        box-shadow: 0 -6px 0 #047857, 0 6px 0 #047857 !important;
    }

    /* Style the native Streamlit Sidebar Close button (inside sidebar) */
    [data-testid="stSidebarCollapseButton"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: auto !important;
    }

    [data-testid="stSidebarCollapseButton"] button,
    section[data-testid="stSidebar"] button[kind="header"],
    section[data-testid="stSidebar"] button[aria-label="Collapse sidebar"] {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 36px !important;
        height: 36px !important;
        background: #ffffff !important;
        color: #065f46 !important;
        border: 1.5px solid #e2e8f0 !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05) !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        pointer-events: auto !important;
    }

    [data-testid="stSidebarCollapseButton"] button:hover,
    section[data-testid="stSidebar"] button[kind="header"]:hover,
    section[data-testid="stSidebar"] button[aria-label="Collapse sidebar"]:hover {
        background: #fee2e2 !important;
        color: #b91c1c !important;
        border-color: #fecaca !important;
        transform: scale(1.05) !important;
    }

    .block-container {
        padding-top: 1.25rem !important;
        padding-bottom: 3rem !important;
        max-width: 1350px !important;
    }

    /* SVG Icon Base Styling */
    .icon {
        display: inline-block;
        vertical-align: middle;
        flex-shrink: 0;
    }

    /* Section Titles with SVG Icons */
    .section-title {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.28rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 1.1rem;
    }
    .section-title svg {
        color: #059669;
    }

    .sub-section-title {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 1.05rem;
        font-weight: 700;
        color: #1e293b;
        margin: 1.3rem 0 0.7rem 0;
    }
    .sub-section-title svg {
        color: #0284c7;
    }

    /* Web App Navbar */
    .app-navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: linear-gradient(135deg, #064e3b 0%, #065f46 50%, #047857 100%);
        padding: 1.25rem 1.75rem;
        border-radius: 16px;
        color: #ffffff;
        margin-bottom: 1.75rem;
        box-shadow: 0 10px 25px -5px rgba(6, 78, 59, 0.3), 0 8px 10px -6px rgba(6, 78, 59, 0.2);
    }
    .brand-section {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .brand-logo {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(8px);
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.25);
    }
    .brand-title {
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0;
        line-height: 1.2;
    }
    .brand-subtitle {
        font-size: 0.82rem;
        color: #a7f3d0;
        margin: 0;
        font-weight: 400;
    }
    .navbar-badges {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(10px);
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #ecfdf5;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .user-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #34d399;
        box-shadow: 0 0 8px #34d399;
    }
    .botany-highlight-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #ffffff;
        color: #065f46 !important;
        padding: 6px 16px;
        border-radius: 9999px;
        font-size: 0.84rem;
        font-weight: 800;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
        border: 2px solid #a7f3d0;
        letter-spacing: 0.02em;
    }
    .botany-hero-tag {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.4);
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
        backdrop-filter: blur(8px);
    }
    .nanstack-nav-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #ffffff;
        color: #0f172a !important;
        text-decoration: none !important;
        padding: 5px 14px 5px 6px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 700;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.6);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
    }
    .nanstack-nav-badge:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
        background: #f8fafc;
        color: #0284c7 !important;
    }
    .nanstack-nav-logo {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        object-fit: contain;
        background: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
    }

    /* Compact Professional Logout Button */
    .navbar-logout-btn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.16);
        backdrop-filter: blur(10px);
        color: #ffffff !important;
        text-decoration: none !important;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        border: 1px solid rgba(255, 255, 255, 0.35);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        white-space: nowrap;
    }
    .navbar-logout-btn:hover {
        background: rgba(239, 68, 68, 0.95);
        border-color: #ef4444;
        color: #ffffff !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 14px rgba(239, 68, 68, 0.4);
    }
    .navbar-logout-btn svg {
        width: 14px;
        height: 14px;
        stroke: currentColor;
    }

    .sidebar-logout-btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        width: 100%;
        background: #fef2f2;
        color: #b91c1c !important;
        text-decoration: none !important;
        padding: 7px 12px;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 700;
        border: 1px solid #fecaca;
        transition: all 0.2s ease;
        cursor: pointer;
        margin-top: 6px;
    }
    .sidebar-logout-btn:hover {
        background: #fee2e2;
        border-color: #fca5a5;
        color: #991b1b !important;
        transform: translateY(-1px);
    }
    .sidebar-logout-btn svg {
        width: 14px;
        height: 14px;
        stroke: currentColor;
    }

    /* KPI Stats Bar */
    .kpi-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-bottom: 1.75rem;
    }
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 14px 18px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
        display: flex;
        flex-direction: column;
        gap: 4px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.04);
        border-color: #cbd5e1;
    }
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
    }
    .kpi-value {
        font-size: 1.45rem;
        font-weight: 800;
        color: #0f172a;
    }
    .kpi-trend {
        font-size: 0.72rem;
        color: #10b981;
        font-weight: 600;
    }

    /* Sidebar Headings & Elements */
    .sidebar-logo-icon {
        background: #059669;
        color: white;
        width: 38px;
        height: 38px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .sidebar-heading {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.92rem;
        font-weight: 700;
        color: #334155;
        margin: 1rem 0 0.5rem 0;
    }
    .sidebar-heading svg {
        color: #059669;
    }

    /* Diagnosis Banner Hero Cards */
    .hero-diag {
        border-radius: 16px;
        padding: 20px 24px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 10px 20px -5px rgba(0, 0, 0, 0.1);
    }
    .hero-healthy {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
    }
    .hero-deficiency {
        background: linear-gradient(135deg, #b91c1c 0%, #dc2626 50%, #ea580c 100%);
    }
    .hero-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
    }
    .hero-title {
        font-size: 1.6rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.01em;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .confidence-badge {
        background: rgba(255, 255, 255, 0.22);
        backdrop-filter: blur(8px);
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.9rem;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    /* Warning Alert Banner */
    .alert-banner {
        background: #fffbeb;
        border-left: 5px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 10px;
        margin-bottom: 18px;
        color: #92400e;
        font-size: 0.9rem;
        line-height: 1.5;
        display: flex;
        align-items: flex-start;
        gap: 12px;
    }
    .alert-banner svg {
        color: #d97706;
        flex-shrink: 0;
        margin-top: 2px;
    }

    /* Modern Disclaimer Card */
    .disclaimer-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #0284c7;
        padding: 12px 16px;
        border-radius: 10px;
        font-size: 0.83rem;
        color: #475569;
        margin-top: 1.25rem;
        display: flex;
        align-items: flex-start;
        gap: 12px;
        line-height: 1.5;
    }
    .disclaimer-card svg {
        color: #0284c7;
        flex-shrink: 0;
        margin-top: 2px;
    }

    /* Modern Footer */
    .app-footer {
        margin-top: 3.5rem;
        padding: 2.2rem 1.5rem;
        background: #ffffff;
        border-top: 1px solid #e2e8f0;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02);
    }
    .footer-flex {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 20px;
    }
    .footer-brand {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .footer-brand-title {
        font-weight: 800;
        font-size: 1.15rem;
        color: #0f172a;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .footer-brand-title svg {
        color: #059669;
    }
    .footer-brand-desc {
        font-size: 0.82rem;
        color: #64748b;
    }
    .footer-nanstack-link {
        text-decoration: none !important;
        display: inline-block;
    }
    .footer-nanstack {
        display: flex;
        align-items: center;
        gap: 12px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 8px 18px;
        border-radius: 9999px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    .nanstack-footer-logo {
        width: 34px;
        height: 34px;
        border-radius: 8px;
        object-fit: contain;
        background: #ffffff;
        padding: 2px;
    }
    .footer-nanstack-text {
        font-size: 0.84rem;
        color: #475569;
    }
    .nanstack-highlight {
        font-weight: 800;
        color: #0f172a;
        font-size: 0.94rem;
        letter-spacing: -0.01em;
    }

    /* Sidebar Footer Credit */
    .sidebar-credit {
        margin-top: 1.5rem;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 16px 14px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .sidebar-logo {
        width: 70px;
        height: 70px;
        object-fit: contain;
        margin: 0 auto 10px auto;
        border-radius: 10px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 4px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        display: block;
    }
    .sidebar-credit-sub {
        font-size: 0.72rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 2px;
    }
    .sidebar-credit-title {
        font-size: 1.1rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 4px;
    }
    .sidebar-credit-desc {
        font-size: 0.78rem;
        color: #0284c7;
        font-weight: 600;
        margin-top: 4px;
    }

    /* ========================================================
       LANDING PAGE & AUTHENTICATION STYLES
       ======================================================== */
    .landing-hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #065f46;
        padding: 6px 16px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 1.25rem;
    }
    .landing-title {
        font-size: 2.7rem;
        font-weight: 800;
        color: #0f172a;
        line-height: 1.15;
        letter-spacing: -0.03em;
        margin: 0 0 1.2rem 0;
    }
    .landing-title-highlight {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .landing-desc {
        font-size: 1.05rem;
        color: #475569;
        line-height: 1.65;
        margin-bottom: 2rem;
    }
    .feature-list {
        display: flex;
        flex-direction: column;
        gap: 16px;
        margin-bottom: 2rem;
    }
    .feature-item {
        display: flex;
        align-items: flex-start;
        gap: 14px;
    }
    .feature-icon-box {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        background: #ecfdf5;
        color: #059669;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
        border: 1px solid #d1fae5;
    }
    .feature-text h4 {
        margin: 0 0 2px 0;
        font-size: 0.96rem;
        font-weight: 700;
        color: #1e293b;
    }
    .feature-text p {
        margin: 0;
        font-size: 0.83rem;
        color: #64748b;
        line-height: 1.45;
    }
    .landing-stats-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 14px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
    }
    .landing-stat {
        text-align: center;
    }
    .landing-stat-val {
        font-size: 1.3rem;
        font-weight: 800;
        color: #0f172a;
        display: block;
    }
    .landing-stat-lbl {
        font-size: 0.68rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .auth-card-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 2.2rem;
        box-shadow: 0 20px 30px -10px rgba(0, 0, 0, 0.07), 0 10px 15px -3px rgba(0, 0, 0, 0.04);
    }
    .auth-header {
        margin-bottom: 1.5rem;
    }
    .auth-header h2 {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0 0 6px 0;
        letter-spacing: -0.02em;
    }
    .auth-header p {
        font-size: 0.88rem;
        color: #64748b;
        margin: 0;
    }
    .demo-credentials-box {
        background: #f8fafc;
        border: 1px dashed #cbd5e1;
        border-radius: 12px;
        padding: 14px;
        margin-top: 1.5rem;
        font-size: 0.8rem;
        color: #475569;
    }

    /* Enhanced Mobile & Tablet Responsiveness */
    button[kind="secondary"], button[kind="primary"], .stButton > button {
        border-radius: 10px !important;
        min-height: 42px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        transition: all 0.2s ease !important;
    }
    
    div[data-baseweb="tab-list"] {
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
        white-space: nowrap !important;
        -webkit-overflow-scrolling: touch;
        gap: 6px !important;
        padding-bottom: 6px !important;
        scrollbar-width: none;
    }
    div[data-baseweb="tab-list"]::-webkit-scrollbar {
        display: none;
    }
    div[data-baseweb="tab"] {
        font-size: 0.85rem !important;
        padding: 8px 12px !important;
        border-radius: 8px !important;
        white-space: nowrap !important;
        flex-shrink: 0 !important;
    }

    @media (max-width: 960px) {
        .kpi-container {
            grid-template-columns: repeat(2, 1fr) !important;
            gap: 10px;
        }
        .landing-title {
            font-size: 2.1rem;
        }
        .landing-stats-row {
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        }
    }

    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
            padding-top: 0.85rem !important;
        }
        div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
            margin-bottom: 0.75rem;
        }
        .app-navbar {
            flex-direction: column;
            align-items: flex-start;
            gap: 12px;
            padding: 1rem;
            border-radius: 12px;
        }
        .brand-section {
            gap: 10px;
        }
        .brand-logo {
            width: 40px;
            height: 40px;
            border-radius: 10px;
        }
        .brand-title {
            font-size: 1.22rem;
        }
        .brand-subtitle {
            font-size: 0.72rem;
        }
        .navbar-badges {
            width: 100%;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
            flex-wrap: wrap;
        }
        .botany-highlight-pill {
            font-size: 0.74rem;
            padding: 4px 10px;
        }
        .botany-hero-tag {
            font-size: 0.72rem;
            padding: 3px 10px;
        }
        .nanstack-nav-badge {
            font-size: 0.75rem;
            padding: 4px 10px 4px 5px;
        }
        .nanstack-nav-logo {
            width: 24px;
            height: 24px;
        }
        .status-pill, .user-pill {
            font-size: 0.72rem;
            padding: 4px 10px;
        }
        .section-title {
            font-size: 1.15rem;
        }
        .hero-diag {
            padding: 16px 18px;
            border-radius: 12px;
        }
        .hero-header {
            flex-direction: column;
            align-items: flex-start;
            gap: 8px;
        }
        .hero-title {
            font-size: 1.25rem;
        }
        .confidence-badge {
            font-size: 0.8rem;
            padding: 4px 10px;
        }
        .disclaimer-card {
            padding: 10px 12px;
            font-size: 0.78rem;
        }
        .app-footer {
            margin-top: 2rem;
            padding: 1.5rem 1rem;
            border-radius: 12px;
        }
        .footer-flex {
            flex-direction: column;
            align-items: flex-start;
            gap: 14px;
        }
        .footer-nanstack-link {
            width: 100%;
        }
        .footer-nanstack {
            width: 100%;
            justify-content: center;
            padding: 8px 14px;
        }
    }

    @media (max-width: 480px) {
        .block-container {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
        .landing-title {
            font-size: 1.65rem;
        }
        .auth-card-box {
            padding: 1.5rem 1.2rem;
            border-radius: 16px;
        }
        .kpi-container {
            grid-template-columns: repeat(2, 1fr) !important;
            gap: 8px !important;
        }
        .kpi-card {
            padding: 8px 10px !important;
        }
        .kpi-label {
            font-size: 0.62rem !important;
        }
        .kpi-value {
            font-size: 1.1rem !important;
        }
        .kpi-trend {
            font-size: 0.6rem !important;
        }
        .hero-title {
            font-size: 1.12rem;
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_predictor():
    """Cached singleton instance of the predictor."""
    return NutrientDeficiencyPredictor()

# =============================================================
# PUBLIC LANDING PAGE & AUTHENTICATION VIEW
# =============================================================
def render_landing_page(uon_logo_b64: str, nanstack_logo_b64: str):
    """
    Renders the public landing page with website introduction,
    feature highlights, research metrics, and the secure login / registration panel.
    """
    # Top Minimal Landing Header
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; align-items:center; padding: 0.5rem 0 1.5rem 0; border-bottom:1px solid #e2e8f0; margin-bottom:2rem; flex-wrap:wrap; gap:12px;">
        <div style="display:flex; align-items:center; gap:14px;">
            <img src="data:image/png;base64,{uon_logo_b64}" style="height:56px; width:auto; object-fit:contain; border-radius:6px;" alt="University of Narowal Logo" />
            <div>
                <h2 style="margin:0; font-size:1.35rem; font-weight:800; color:#0f172a; letter-spacing:-0.02em; line-height:1.2;">Department Of Botany</h2>
                <p style="margin:2px 0 0 0; font-size:0.92rem; font-weight:700; color:#059669; letter-spacing:0.01em;">University Of Narowal</p>
            </div>
        </div>
        <div>
            <div style="display:inline-flex; align-items:center; gap:8px; background:#ecfdf5; border:1.5px solid #a7f3d0; color:#065f46; padding:6px 16px; border-radius:9999px; font-size:0.84rem; font-weight:800; box-shadow:0 2px 6px rgba(5,150,105,0.08);">
                <span>🌱</span>
                <span>AgriVision AI Platform</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Main 2-Column Landing Section
    col_intro, col_auth = st.columns([1.25, 1], gap="large")

    # LEFT COLUMN: Introduction & Feature Showcase
    with col_intro:
        st.markdown(f"""
        <div class="landing-hero-badge">
            <span class="pulse-dot"></span>
            <span>Precision Crop Health AI</span>
        </div>
        <h1 class="landing-title">
            AI Based Plant Nutrients <span class="landing-title-highlight">Deficiency Detection</span>
        </h1>
        <p class="landing-desc">
            AgriVision AI is an end-to-end computer vision and deep learning platform designed to provide agronomists, 
            researchers, and growers with rapid, laboratory-grade nutritional screening from smartphone leaf imagery.
        </p>
        """, unsafe_allow_html=True)

        # Core Value Pillars (Cleaned)
        st.markdown(f"""
        <div class="feature-list">
            <div class="feature-item">
                <div class="feature-icon-box">{ICONS['layers']}</div>
                <div class="feature-text">
                    <h4>Macro & Micro Nutrients Screening (9 Target Classes)</h4>
                    <p>Detects deficiencies across macronutrients and micronutrients including Nitrogen, Phosphorus, Potassium, Calcium, Iron, Magnesium, Sulphur, Zinc, and Healthy foliage.</p>
                </div>
            </div>
            <div class="feature-item">
                <div class="feature-icon-box">{ICONS['clipboard']}</div>
                <div class="feature-text">
                    <h4>Actionable Agronomic Guidance (English & Urdu)</h4>
                    <p>Instant corrective treatment recommendations, fertilizer dosages (Urea, DAP, MOP, Chelated Iron), and soil pH management guidelines.</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Research Statistics Bar
        st.markdown("""
        <div class="landing-stats-row">
            <div class="landing-stat">
                <span class="landing-stat-val">96.85%</span>
                <span class="landing-stat-lbl">Test Accuracy</span>
            </div>
            <div class="landing-stat">
                <span class="landing-stat-val">10</span>
                <span class="landing-stat-lbl">Classes</span>
            </div>
            <div class="landing-stat">
                <span class="landing-stat-val">99,480</span>
                <span class="landing-stat-lbl">Curated Leaves</span>
            </div>
            <div class="landing-stat">
                <span class="landing-stat-val">&lt; 0.18s</span>
                <span class="landing-stat-lbl">Latency</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # RIGHT COLUMN: Authentication Card
    with col_auth:
        st.markdown('<div class="auth-card-box">', unsafe_allow_html=True)
        
        # Segmented Control / Tabs for Auth Mode
        auth_mode = st.radio(
            "Auth Mode",
            ["Sign In", "Create Account", "Forgot Password"],
            horizontal=True,
            label_visibility="collapsed"
        )

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

        # ---------------------------------------------
        # MODE 1: SIGN IN
        # ---------------------------------------------
        if auth_mode == "Sign In":
            st.markdown("""
            <div class="auth-header">
                <h2>Welcome Back</h2>
                <p>Sign in to access the protected diagnostic workstation.</p>
            </div>
            """, unsafe_allow_html=True)

            with st.form("form_signin", clear_on_submit=False):
                username_or_email = st.text_input(
                    "Email or Username",
                    placeholder="Enter your username or email",
                    value=""
                )
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter your password",
                    value=""
                )
                
                col_rem, col_fill = st.columns([1, 1])
                with col_rem:
                    remember_me = st.checkbox("Remember this session", value=True)
                with col_fill:
                    st.caption("Secure PBKDF2-SHA256")

                submit_login = st.form_submit_button("Sign In to Workstation", use_container_width=True)

                if submit_login:
                    success, user_data, message = authenticate_user(username_or_email, password)
                    if success:
                        st.session_state["authenticated"] = True
                        st.session_state["user"] = user_data
                        st.success(f"Welcome back, {user_data['full_name']}!")
                        st.rerun()
                    else:
                        st.error(f"Authentication Error: {message}")

        # ---------------------------------------------
        # MODE 2: CREATE ACCOUNT
        # ---------------------------------------------
        elif auth_mode == "Create Account":
            st.markdown("""
            <div class="auth-header">
                <h2>Create Account</h2>
                <p>Register to deploy leaf diagnostics in your agricultural field.</p>
            </div>
            """, unsafe_allow_html=True)

            with st.form("form_register", clear_on_submit=False):
                reg_name = st.text_input("Full Name", placeholder="e.g. Dr. Emily Chen")
                reg_username = st.text_input("Username", placeholder="e.g. emily_agri")
                reg_email = st.text_input("Email Address", placeholder="e.g. emily@research.edu")
                reg_password = st.text_input("Password", type="password", placeholder="Minimum 6 characters")

                submit_reg = st.form_submit_button("Create Account", use_container_width=True)

                if submit_reg:
                    success, message = register_user(reg_username, reg_email, reg_name, reg_password)
                    if success:
                        st.success(message)
                        st.info("You can now switch to 'Sign In' to access your account.")
                    else:
                        st.error(message)

        # ---------------------------------------------
        # MODE 3: FORGOT PASSWORD
        # ---------------------------------------------
        elif auth_mode == "Forgot Password":
            st.markdown("""
            <div class="auth-header">
                <h2>Reset Password</h2>
                <p>Enter your registered email address and choose a new password.</p>
            </div>
            """, unsafe_allow_html=True)

            with st.form("form_reset", clear_on_submit=False):
                reset_email = st.text_input("Registered Email Address", placeholder="e.g. admin@agrivision.ai")
                new_pwd = st.text_input("New Password", type="password", placeholder="Enter new password (min 6 chars)")

                submit_reset = st.form_submit_button("Update Password", use_container_width=True)

                if submit_reset:
                    success, message = reset_password(reset_email, new_pwd)
                    if success:
                        st.success(message)
                        st.info("Switch to 'Sign In' with your updated credentials.")
                    else:
                        st.error(message)

        st.markdown('</div>', unsafe_allow_html=True)

    # Landing Page Footer (Company name once at page bottom, no link)
    st.markdown(f"""
    <div style="margin-top:3rem; padding:1.5rem; border-top:1px solid #e2e8f0; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; font-size:0.85rem; color:#64748b;">
        <div>© 2026 <b>Department of Botany</b> • AgriVision AI Platform</div>
        <div style="font-weight:600; color:#475569;">
            Engineered & Developed by <span style="color:#0f172a; font-weight:700;">Nanstack</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# =============================================================
# PROTECTED WORKSTATION & DASHBOARD VIEW
# =============================================================
def render_protected_dashboard(uon_logo_b64: str, nanstack_logo_b64: str, user: dict):
    """
    Renders the full protected AgriVision AI diagnostic workstation,
    including the leaf analysis module, Grad-CAM XAI visualizer,
    research evaluation metrics, and nutrient encyclopedia.
    """
    predictor = get_predictor()
    # Initialize language preference from session state early
    is_urdu = st.session_state.get("lang_choice", "English") == "اردو (Urdu)"

    # =========================================================
    # SIDEBAR CONTROLS
    # =========================================================
    with st.sidebar:
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <div style="display:flex; align-items:center; gap:12px;">
                <img src="data:image/png;base64,{uon_logo_b64}" style="height:44px; width:auto; object-fit:contain; border-radius:4px;" alt="University of Narowal Logo" />
                <div>
                    <h3 style="margin:0; font-size:1.05rem; font-weight:800; color:#0f172a; line-height:1.2;">Department Of Botany</h3>
                    <p style="margin:2px 0 0 0; font-size:0.8rem; font-weight:700; color:#059669;">University Of Narowal</p>
                </div>
            </div>
            <button onclick="
                var doc = (window.parent && window.parent.document) ? window.parent.document : document;
                var btn = doc.querySelector('[data-testid=stSidebarCollapseButton] button, [data-testid=stSidebarCollapseButton], button[aria-label*=Collapse]');
                if (!btn) btn = document.querySelector('[data-testid=stSidebarCollapseButton] button, [data-testid=stSidebarCollapseButton], button[aria-label*=Collapse]');
                if (btn) btn.click();
            " style="background:#f1f5f9; border:1px solid #cbd5e1; color:#334155; padding:5px 10px; border-radius:8px; font-size:0.75rem; font-weight:700; cursor:pointer; display:inline-flex; align-items:center; gap:4px;" title="Close Sidebar">
                ✕ <span>{'بند کریں' if is_urdu else 'Close'}</span>
            </button>
        </div>
        """, unsafe_allow_html=True)

        # Authenticated User Badge
        st.markdown(f"""
        <div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px; padding:10px 12px; margin-bottom:14px;">
            <div style="font-size:0.7rem; text-transform:uppercase; color:#166534; font-weight:700; letter-spacing:0.04em;">Signed In As</div>
            <div style="font-size:0.88rem; font-weight:800; color:#0f172a; margin-top:2px;">{user.get('full_name', 'Agronomist')}</div>
            <div style="font-size:0.75rem; color:#059669; margin-bottom:8px;">Role: {user.get('role', 'agronomist').title()}</div>
            <a href="?logout=true" target="_self" class="sidebar-logout-btn">
                {ICONS['log_out']}
                <span>{'لاگ آؤٹ' if is_urdu else 'Log Out'}</span>
            </a>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Language Selection
        st.markdown(f"""
        <div class="sidebar-heading">
            {ICONS['globe']}
            <span>{'زبان منتخب کریں' if 'اردو' in st.session_state.get('lang_choice', '') else 'Interface Language'}</span>
        </div>
        """, unsafe_allow_html=True)
        
        lang = st.radio(
            "Language",
            ["English", "اردو (Urdu)"],
            index=0,
            key="lang_choice",
            label_visibility="collapsed"
        )
        is_urdu = lang == "اردو (Urdu)"

        # Target Crop Filter
        st.markdown(f"""
        <div class="sidebar-heading">
            {ICONS['layers']}
            <span>{'فصل کا دائرہ' if is_urdu else 'Crop Scope'}</span>
        </div>
        """, unsafe_allow_html=True)
        
        crop_selected = st.selectbox(
            "Crop Selection",
            ["Multi-Crop (Universal)", "Banana (Musa spp.)", "Rice (Oryza sativa)"],
            label_visibility="collapsed"
        )

        st.divider()

        # System Architecture Status Card
        st.markdown(f"""
        <div class="sidebar-heading">
            {ICONS['cpu']}
            <span>{'نظام کی حالت' if is_urdu else 'Engine Status'}</span>
        </div>
        """, unsafe_allow_html=True)
        
        if predictor.is_ready():
            st.success(f"● **Backbone:** {predictor.architecture.upper()}")
            st.caption(f"**Target Classes:** {len(predictor.class_names)} Categories")
            if predictor.val_acc > 0:
                st.caption(f"**Validation Accuracy:** {predictor.val_acc:.1%}")
        else:
            st.warning("Model weights not loaded.")

        # Sidebar Nanstack Credit Card with Logo (No link, name is enough)
        st.markdown(f"""
        <div class="sidebar-credit">
            <img src="data:image/jpeg;base64,{nanstack_logo_b64}" class="sidebar-logo" alt="Nanstack" />
            <div class="sidebar-credit-sub">Engineered & Maintained by</div>
            <div class="sidebar-credit-title">Nanstack</div>
        </div>
        """, unsafe_allow_html=True)


    # =========================================================
    # APP NAVIGATION HEADER / HERO BANNER (BOTANY DEPARTMENT)
    # =========================================================
    user_display_name = user.get("full_name", "Agronomist").split()[0]
    
    st.markdown(f"""
    <div class="app-navbar" style="margin-bottom:1.5rem;">
        <div class="brand-section" style="display:flex; align-items:center; gap:14px;">
            <img src="data:image/png;base64,{uon_logo_b64}" style="height:54px; width:auto; object-fit:contain; background:#ffffff; border-radius:8px; padding:3px; box-shadow:0 2px 6px rgba(0,0,0,0.08);" alt="University of Narowal Logo" />
            <div>
                <h1 class="brand-title" style="margin:0; font-size:1.35rem; font-weight:800; color:#ffffff; line-height:1.2;">Department Of Botany</h1>
                <p class="brand-subtitle" style="margin:2px 0 0 0; font-size:0.88rem; color:#a7f3d0; font-weight:700;">University Of Narowal</p>
            </div>
        </div>
        <div class="navbar-badges">
            <div class="botany-highlight-pill">
                <span>🌱</span>
                <span>AgriVision AI</span>
            </div>
            <a href="?logout=true" target="_self" class="navbar-logout-btn">
                {ICONS['log_out']}
                <span>{'لاگ آؤٹ' if is_urdu else 'Log Out'}</span>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # =========================================================
    # KPI OVERVIEW STATS ROW
    # =========================================================
    kpi_html = f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <span class="kpi-label">{'ماڈل کی درستگی' if is_urdu else 'Test Set Accuracy'}</span>
            <span class="kpi-value">96.85%</span>
            <span class="kpi-trend">{'10 زمرہ جات پر جانچا گیا' if is_urdu else 'Evaluated on 10 Held-Out Classes'}</span>
        </div>
        <div class="kpi-card">
            <span class="kpi-label">{'غذائی زمرہ جات' if is_urdu else 'Target & Rejection Classes'}</span>
            <span class="kpi-value">10 Classes</span>
            <span class="kpi-trend">{'9 غذائی کمی + غیر متعلقہ ڈیٹا فلٹر' if is_urdu else '9 Nutrients + Irrelevant Filter (99.8% F1)'}</span>
        </div>
        <div class="kpi-card">
            <span class="kpi-label">{'تربیت یافتہ تصاویر' if is_urdu else 'Curated Dataset'}</span>
            <span class="kpi-value">99,480</span>
            <span class="kpi-trend">{'99 ہزار سے زائد پتوں کا ڈیٹاسیٹ' if is_urdu else 'Balanced Foliar & OOD Splits'}</span>
        </div>
        <div class="kpi-card">
            <span class="kpi-label">{'تشخیصی وقت' if is_urdu else 'Inference Latency'}</span>
            <span class="kpi-value">&lt; 0.18s</span>
            <span class="kpi-trend">Real-Time MobileNetV2 Inference</span>
        </div>
    </div>
    """
    st.markdown(kpi_html, unsafe_allow_html=True)

    # =========================================================
    # TABS NAVIGATION
    # =========================================================
    tab_diag, tab_encyclopedia = st.tabs([
        "Leaf Diagnostic & Saliency Analysis" if not is_urdu else "پتوں کی تشخیص اور تجزیہ",
        "Nutrient Deficiency Encyclopedia" if not is_urdu else "غذائی اجزاء کی معلوماتی گائیڈ"
    ])

    # ---------------------------------------------------------
    # TAB 1: INTERACTIVE DIAGNOSIS & EXPLAINABILITY
    # ---------------------------------------------------------
    with tab_diag:
        col_left, col_right = st.columns([1, 1.25], gap="large")

        with col_left:
            st.markdown(f"""
            <div class="section-title">
                {ICONS['upload']}
                <span>{'پتے کی تصویر فراہم کریں' if is_urdu else 'Input Leaf Image'}</span>
            </div>
            """, unsafe_allow_html=True)
            
            input_mode = st.radio(
                "Source" if not is_urdu else "ذریعہ",
                ["Upload File" if not is_urdu else "تصویر اپ لوڈ کریں", "Camera Capture" if not is_urdu else "کیمرہ کھولیں"],
                horizontal=True,
                label_visibility="collapsed"
            )

            image_file = None
            if "Upload" in input_mode or "اپ لوڈ" in input_mode:
                image_file = st.file_uploader(
                    "Drop leaf photo here" if not is_urdu else "یہاں پتے کی تصویر ڈراپ کریں",
                    type=["jpg", "jpeg", "png"]
                )
            else:
                image_file = st.camera_input("Capture with device camera" if not is_urdu else "کیمرے سے تصویر لیں")

            # One-Click Quick Presets for Demo
            st.markdown(f"**{'یا فوری معائنے کے لیے نمونہ منتخب کریں:' if is_urdu else 'Or test immediately with verified reference samples:'}**")
            p_col1, p_col2, p_col3, p_col4 = st.columns(4)

            presets = [
                ("Healthy Leaf", ROOT_DIR / "app/assets/samples/healthy_sample.jpg" if (ROOT_DIR / "app/assets/samples/healthy_sample.jpg").exists() else ROOT_DIR / "data/processed/test/Healthy/Healthy_0000.jpg", "Healthy Leaf (Reference)"),
                ("Nitrogen (N)", ROOT_DIR / "app/assets/samples/nitrogen_sample.jpg" if (ROOT_DIR / "app/assets/samples/nitrogen_sample.jpg").exists() else ROOT_DIR / "data/processed/test/Nitrogen_N/Nitrogen_N_0000.jpg", "Nitrogen (N) Chlorosis"),
                ("Potassium (K)", ROOT_DIR / "app/assets/samples/potassium_sample.jpg" if (ROOT_DIR / "app/assets/samples/potassium_sample.jpg").exists() else ROOT_DIR / "data/processed/test/Potassium_K/Potassium_K_0000.jpg", "Potassium (K) Scorch"),
                ("Irrelevant Data", ROOT_DIR / "app/assets/samples/irrelevant_sample.jpg" if (ROOT_DIR / "app/assets/samples/irrelevant_sample.jpg").exists() else ROOT_DIR / "data/processed/test/Irrelevant_Non_Plant/Irrelevant_Non_Plant_test_0000.jpg", "Irrelevant (Non-Plant)"),
            ]

            for idx, (label, p_path, display_name) in enumerate(presets):
                target_col = [p_col1, p_col2, p_col3, p_col4][idx]
                with target_col:
                    if st.button(display_name, key=f"btn_preset_{idx}", use_container_width=True):
                        st.session_state["active_sample_path"] = str(p_path)

            if "active_sample_path" in st.session_state and not image_file:
                sample_p = Path(st.session_state["active_sample_path"])
                if sample_p.exists():
                    image_file = sample_p

            uploaded_img = None
            is_valid_format = True
            format_err_msg = None

            if image_file:
                try:
                    if hasattr(image_file, "seek"):
                        image_file.seek(0)
                    uploaded_img = Image.open(image_file).convert("RGB")
                    st.image(
                        uploaded_img,
                        caption="Source Image Loaded" if not is_urdu else "فراہم کردہ پتے کی تصویر",
                        width="stretch"
                    )
                except Exception as img_err:
                    uploaded_img = None
                    is_valid_format = False
                    format_err_msg = str(img_err)
                    st.markdown(f"""
                    <div style="background:#fef2f2; border:1px solid #fecaca; border-radius:10px; padding:14px; margin-top:10px;">
                        <b style="color:#991b1b;">{'فائل فارمیٹ غلط ہے' if is_urdu else 'Invalid File Format'}</b>
                        <p style="margin:4px 0 0 0; color:#b91c1c; font-size:0.85rem;">
                            {'یہ فائل تصویری فارمیٹ (JPEG/PNG/WEBP) میں نہیں ہے یا خراب ہے۔' if is_urdu else 'The uploaded file is not a valid image file (JPEG/PNG/WEBP) or is corrupted.'}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

        with col_right:
            st.markdown(f"""
            <div class="section-title">
                {ICONS['microscope']}
                <span>{'تشخیصی رپورٹ اور علاج' if is_urdu else 'Diagnostic Report & Agronomic Prescription'}</span>
            </div>
            """, unsafe_allow_html=True)

            if image_file:
                if not predictor.is_ready():
                    st.error("Model weights not available. Please train the model first.")
                elif not is_valid_format or uploaded_img is None:
                    # Invalid format rejection
                    st.markdown(f"""
                    <div style="background:#fef2f2; border:1px solid #fecaca; border-radius:14px; padding:22px; margin-bottom:1.5rem;">
                        <div style="display:flex; align-items:flex-start; gap:14px;">
                            <div style="color:#dc2626; margin-top:2px;">{ICONS['alert_triangle']}</div>
                            <div>
                                <h3 style="margin:0 0 6px 0; color:#991b1b; font-size:1.15rem; font-weight:800;">
                                    {'ناقابلِ قبول فائل فارمیٹ' if is_urdu else 'Unsupported or Corrupted File Format'}
                                </h3>
                                <p style="margin:0 0 10px 0; color:#b91c1c; font-size:0.9rem; line-height:1.5;">
                                    {'سسٹم صرف درست تصاویر (JPEG, PNG, WEBP) قبول کرتا ہے۔ غیر تصویری فائلیں یا خراب ڈیٹا پر کوئی پیش گوئی نہیں کی جائے گی۔' if is_urdu else
                                     'AgriVision AI strictly requires genuine image formats (JPEG, PNG, or WEBP). The uploaded file could not be decoded as an authentic image.'}
                                </p>
                                <div style="display:inline-block; background:#fee2e2; border-radius:8px; padding:6px 12px; font-size:0.8rem; color:#7f1d1d; font-weight:700;">
                                    {'پیش گوئی روک دی گئی ہے' if is_urdu else 'Diagnostic Pipeline Halted • No Prediction Made'}
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    with st.spinner("Executing neural inference & validating domain..." if not is_urdu else "ڈیپ لرننگ ماڈل سے تصدیق اور تجزیہ ہو رہا ہے..."):
                        if hasattr(image_file, "seek"):
                            image_file.seek(0)
                        res = predictor.predict(uploaded_img if uploaded_img is not None else image_file, generate_cam=True)

                    if res.get("status") == "rejected":
                        # STRICT REJECTION: PREDICT NOTHING
                        rejection_type = res.get("rejection_type", "irrelevant_data")
                        rej_msg = res.get("rejection_message_ur" if is_urdu else "rejection_message")

                        if rejection_type == "invalid_format":
                            st.markdown(f"""
                            <div style="background:#fef2f2; border:1px solid #fecaca; border-radius:14px; padding:22px; margin-bottom:1.5rem;">
                                <div style="display:flex; align-items:flex-start; gap:14px;">
                                    <div style="color:#dc2626; margin-top:2px;">{ICONS['alert_triangle']}</div>
                                    <div>
                                        <h3 style="margin:0 0 6px 0; color:#991b1b; font-size:1.15rem; font-weight:800;">
                                            {'ناقابلِ قبول فائل فارمیٹ' if is_urdu else 'Invalid or Corrupted File Format'}
                                        </h3>
                                        <p style="margin:0 0 10px 0; color:#b91c1c; font-size:0.9rem; line-height:1.5;">
                                            {rej_msg}
                                        </p>
                                        <div style="display:inline-block; background:#fee2e2; border-radius:8px; padding:6px 12px; font-size:0.8rem; color:#7f1d1d; font-weight:700;">
                                            {'پیش گوئی روک دی گئی ہے' if is_urdu else 'Diagnostic Pipeline Halted • No Prediction Made'}
                                        </div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style="background:#fffbeb; border:1px solid #fde68a; border-radius:14px; padding:22px; margin-bottom:1.5rem;">
                                <div style="display:flex; align-items:flex-start; gap:14px;">
                                    <div style="color:#d97706; margin-top:2px;">{ICONS['shield']}</div>
                                    <div>
                                        <h3 style="margin:0 0 6px 0; color:#92400e; font-size:1.15rem; font-weight:800;">
                                            {'غیر متعلقہ تصویر (پودوں سے ہٹ کر)' if is_urdu else 'Irrelevant / Non-Plant Data Detected'}
                                        </h3>
                                        <p style="margin:0 0 10px 0; color:#b45309; font-size:0.9rem; line-height:1.5;">
                                            {rej_msg}
                                        </p>
                                        <div style="display:inline-block; background:#fef3c7; border:1px solid #fde68a; border-radius:8px; padding:6px 12px; font-size:0.8rem; color:#78350f; font-weight:700;">
                                            {'ڈومین فلٹر فعال: کوئی پیش گوئی نہیں کی گئی' if is_urdu else 'Domain Enforcement Active • All Predictions Suppressed'}
                                        </div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        # STATUS IS SUCCESS: RENDER FULL DIAGNOSTIC REPORT
                        top_class = res["predicted_class"]
                        confidence = res["confidence"]
                        is_confident = res["is_confident"]
                        guidance = res["guidance"]
                        is_healthy = "healthy" in top_class.lower()

                        # Hero Result Card with Vector SVG Icons
                        hero_cls = "hero-healthy" if is_healthy else "hero-deficiency"
                        status_icon = ICONS["check_circle"] if is_healthy else ICONS["alert_triangle"]
                        class_title = guidance.get("title_ur" if is_urdu else "title", top_class)

                        hero_html = f"""
                        <div class="hero-diag {hero_cls}">
                            <div class="hero-header">
                                <div>
                                    <span style="font-size:0.78rem; text-transform:uppercase; letter-spacing:0.06em; opacity:0.92;">
                                        {'تجزیاتی نتیجہ' if is_urdu else 'Primary Diagnosis'}
                                    </span>
                                    <h2 class="hero-title">{status_icon} <span>{class_title}</span></h2>
                                </div>
                                <div class="confidence-badge">
                                    {'اعتماد:' if is_urdu else 'Confidence:'} {confidence:.1%}
                                </div>
                            </div>
                        </div>
                        """
                        st.markdown(hero_html, unsafe_allow_html=True)

                        # Uncertainty / Blur Alert
                        if not is_confident:
                            st.markdown(f"""
                            <div class="alert-banner">
                                {ICONS['alert_triangle']}
                                <div>
                                    <b>{'توجہ فرمائیں: کم اعتماد یا غیر واضح تصویر' if is_urdu else 'Notice: Low Diagnostic Confidence or Blur Detected'}</b><br>
                                    {'تصویر کا معیار کم ہے یا ماڈل کا اعتماد 55% سے نیچے ہے۔ درست نتائج کے لیے مناسب روشنی میں واضح تصویر لیں۔' if is_urdu else
                                     'The prediction confidence is below the diagnostic threshold (55%) or the leaf is out of focus. For definitive diagnosis, please upload a clearer image.'}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Differential Probabilities
                        st.markdown(f"""
                        <div class="sub-section-title">
                            {ICONS['bar_chart']}
                            <span>{'ممکنہ متبادل زمرہ جات' if is_urdu else 'Differential Probabilities'}</span>
                        </div>
                        """, unsafe_allow_html=True)

                        for pred in res["top_predictions"]:
                            p_name = pred["class"].replace("_", " ")
                            p_conf = pred["confidence"]
                            st.write(f"**{p_name}** — `{p_conf:.1%}`")
                            st.progress(float(p_conf))

                        st.divider()

                        # Explainable AI (Grad-CAM) Visualizer
                        st.markdown(f"""
                        <div class="sub-section-title">
                            {ICONS['scan']}
                            <span>{'قابلِ وضاحت اے آئی (Grad-CAM معائنہ)' if is_urdu else 'Explainable AI: Grad-CAM Saliency Map'}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        st.caption(
                            "Gradient-weighted activation map showing spatial regions (chlorotic tissue, margin scorch, veins) driving the diagnosis:" if not is_urdu else
                            "گریڈینٹ ہیٹ میپ واضح کرتا ہے کہ ماڈل نے پتے کے کن حصوں (رگوں، کناروں یا دھبوں) سے کمی کی تشخیص کی:"
                        )

                        if res.get("gradcam_overlay") is not None:
                            cam_col1, cam_col2 = st.columns(2)
                            with cam_col1:
                                st.image(uploaded_img, caption="Input Leaf Image", width="stretch")
                            with cam_col2:
                                st.image(res["gradcam_overlay"], caption="Grad-CAM Saliency Map", width="stretch")

                        st.divider()

                        # Agronomic Prescription
                        st.markdown(f"""
                        <div class="sub-section-title">
                            {ICONS['clipboard']}
                            <span>{'زرعی رہنمائی اور تدارک کا منصوبہ' if is_urdu else 'Agronomic Action Plan'}</span>
                        </div>
                        """, unsafe_allow_html=True)

                        symptoms_list = guidance.get("symptoms_ur" if is_urdu else "symptoms", [])
                        remedies_list = guidance.get("remedies", [])
                        fertilizer = guidance.get("fertilizer_recommendation", "N/A")
                        soil_factors = guidance.get("soil_factors", "N/A")

                        with st.expander("Identified Visual Symptoms" if not is_urdu else "پتوں کی ظاہری علامات", expanded=True):
                            for sym in symptoms_list:
                                st.markdown(f"• {sym}")

                        with st.expander("Prescription: Fertilizers & Remediation" if not is_urdu else "تجویز کردہ کھاد اور علاج", expanded=True):
                            st.markdown(f"**{'تجویز کردہ کھاد:' if is_urdu else 'Recommended Formulation:'}** `{fertilizer}`")
                            st.markdown(f"**{'مٹی اور پی ایچ کا انتظام:' if is_urdu else 'Soil pH & Root Management:'}** {soil_factors}")
                            st.markdown(f"**{'اقدامات:' if is_urdu else 'Management Actions:'}**")
                            for rem in remedies_list:
                                st.markdown(f"- {rem}")

                        # Professional Disclaimer Card with Shield Icon
                        disclaimer = res["disclaimer_ur"] if is_urdu else res["disclaimer_en"]
                        st.markdown(f"""
                        <div class="disclaimer-card">
                            {ICONS['shield']}
                            <div><b>Advisory Disclaimer:</b> {disclaimer}</div>
                        </div>
                        """, unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div style="background:#f8fafc; border:2px dashed #cbd5e1; border-radius:16px; padding:45px 20px; text-align:center; color:#64748b;">
                    <div style="margin-bottom:12px;">{ICONS['upload_large']}</div>
                    <h4 style="margin:0 0 6px 0; color:#334155; font-size:1.1rem;">{'کوئی تصویر منتخب نہیں ہوئی' if is_urdu else 'No Leaf Image Selected'}</h4>
                    <p style="margin:0; font-size:0.88rem;">{'بائیں جانب سے تصویر اپ لوڈ کریں یا فوری ٹیسٹ کے لیے نمونہ پر کلک کریں۔' if is_urdu else 'Upload a photo or tap one of the reference sample buttons to trigger the diagnostic pipeline.'}</p>
                </div>
                """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # TAB 2: AGRONOMIC ENCYCLOPEDIA
    # ---------------------------------------------------------
    with tab_encyclopedia:
        st.markdown(f"""
        <div class="section-title">
            {ICONS['book']}
            <span>{'پودوں کے غذائی اجزاء کی جامع گائیڈ' if is_urdu else 'Comprehensive Nutrient Deficiency Reference Guide'}</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("Standard agronomic reference guide for all macronutrients and micronutrients covered by the AI system:")

        for nutrient_key, info in NUTRIENT_KNOWLEDGE_BASE.items():
            title_display = f"{info['title']} / {info.get('title_ur', '')}" if is_urdu else info['title']
            with st.expander(f"{title_display}", expanded=False):
                e1, e2 = st.columns(2)
                with e1:
                    st.markdown("**Diagnostic Visual Symptoms:**")
                    for s in info["symptoms"]:
                        st.markdown(f"• {s}")
                    st.markdown("**Physiological Causes:**")
                    for c in info["causes"]:
                        st.markdown(f"• {c}")
                with e2:
                    st.markdown(f"**Fertilizer Recommendations:**\n• {info['fertilizer_recommendation']}")
                    st.markdown(f"**Soil Dynamics & pH:**\n• {info['soil_factors']}")
                    st.markdown("**Corrective Remediation:**")
                    for r in info["remedies"]:
                        st.markdown(f"- {r}")

    # =========================================================
    # APP FOOTER WITH BOTANY DEPT & NANSTACK BRANDING
    # =========================================================
    footer_html = f"""
    <div class="app-footer">
        <div class="footer-flex">
            <div class="footer-brand">
                <span class="footer-brand-title">
                    {ICONS['leaf']} AgriVision AI • <b>Department of Botany</b>
                    <span style="font-size:0.75rem; color:#059669; font-weight:700; background:#ecfdf5; padding:2px 8px; border-radius:9999px;">Production Web App</span>
                </span>
                <span class="footer-brand-desc">{'شعبہ نباتیات • مصنوعی ذہانت اور ڈیپ لرننگ کے ذریعے پودوں کی غذائی کمی کی خودکار تشخیص' if is_urdu else 'Department of Botany • Deep Learning & Computer Vision for Precision Crop Nutrition'}</span>
            </div>
            <div class="footer-nanstack">
                <img src="data:image/jpeg;base64,{nanstack_logo_b64}" class="nanstack-footer-logo" alt="Nanstack Logo" />
                <div style="display:flex; flex-direction:column; align-items:flex-start; text-align:left;">
                    <span class="footer-nanstack-text">Engineered & Developed by <span class="nanstack-highlight">Nanstack</span></span>
                </div>
            </div>
        </div>
        <div style="margin-top:1.25rem; padding-top:1rem; border-top:1px solid #f1f5f9; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; font-size:0.75rem; color:#94a3b8;">
            <span>© 2026 AgriVision System • Department of Botany</span>
            <span>AI Based Plant Nutrients Deficiency Detection</span>
        </div>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

# =============================================================
# MAIN CONTROLLER
# =============================================================
def main():
    uon_logo_b64 = get_base64_image(UON_LOGO_PATH)
    nanstack_logo_b64 = get_base64_image(NANSTACK_LOGO_PATH)

    # Handle query param logout action
    if st.query_params.get("logout") == "true":
        st.session_state["authenticated"] = False
        st.session_state["user"] = None
        st.query_params.clear()
        st.rerun()

    # Initialize Session State for Authentication
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
    if "user" not in st.session_state:
        st.session_state["user"] = None

    # Routing based on Authentication State
    if not st.session_state["authenticated"]:
        render_landing_page(uon_logo_b64, nanstack_logo_b64)
    else:
        render_protected_dashboard(uon_logo_b64, nanstack_logo_b64, st.session_state["user"])

if __name__ == "__main__":
    main()
