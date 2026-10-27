"""Visual identity: colors, fonts and the Qt style sheet."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QColor, QFont, QFontDatabase

FONTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
CHECK_SVG = (FONTS_DIR.parent / "check.svg").as_posix()

# Palette
BG = "#0B0D11"          # app ground
STAGE = "#07090C"       # behind the video
PANEL = "#0F1217"       # side panels, bars
CARD = "#161A21"        # cards inside panels
CARD_2 = "#12161C"
RAISED = "#1A1F28"
ACTIVE = "#1E2430"
LINE = "#232833"        # panel borders
LINE_SOFT = "#1B2029"
BORDER = "#2F3746"      # control borders
TEXT = "#E8EAED"
TEXT_2 = "#C9CED6"
MUTED = "#A3ABB8"
FAINT = "#8B94A3"
DIM = "#7D8696"
ACCENT = "#C6F24E"      # the single action color
ACCENT_HOVER = "#D4FA6A"
ON_ACCENT = "#0F1115"
OK = "#4ADE80"
WARN = "#FB923C"
WARN_TEXT = "#FDBA74"
DANGER = "#F87171"

RIGHT = "#2DD4BF"       # body right side
LEFT = "#A78BFA"        # body left side
CENTER = "#E8EAED"

SANS = "IBM Plex Sans"
MONO = "IBM Plex Mono"
DISPLAY = "Space Grotesk"


def palette():
    """Dark palette for every Qt widget, including standard dialogs (open file, colors,
    messages) that the style sheet does not fully cover."""
    from PySide6.QtGui import QPalette
    pal = QPalette()
    roles = {
        QPalette.Window: PANEL, QPalette.WindowText: TEXT, QPalette.Base: CARD_2,
        QPalette.AlternateBase: "#11151B", QPalette.ToolTipBase: RAISED, QPalette.ToolTipText: TEXT,
        QPalette.PlaceholderText: DIM, QPalette.Text: TEXT, QPalette.Button: CARD,
        QPalette.ButtonText: TEXT, QPalette.BrightText: "#FFFFFF", QPalette.Highlight: "#3B4A1A",
        QPalette.HighlightedText: TEXT, QPalette.Link: ACCENT, QPalette.LinkVisited: ACCENT_HOVER,
        QPalette.Light: "#2A303A", QPalette.Midlight: "#222834", QPalette.Mid: LINE,
        QPalette.Dark: "#0A0C10", QPalette.Shadow: "#000000",
    }
    for role, color in roles.items():
        pal.setColor(role, QColor(color))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, QColor(DIM))
    return pal


def load_fonts():
    for f in FONTS_DIR.glob("*.ttf"):
        QFontDatabase.addApplicationFont(str(f))


# fonts with the letters of Japanese, Korean and Chinese, in the order of each language (the same
# character is drawn differently in each); the rest of the text keeps the app's own fonts
CJK = {"ja": ["Yu Gothic UI", "Meiryo UI", "Noto Sans CJK JP", "Noto Sans JP"],
       "ko": ["Malgun Gothic", "Noto Sans CJK KR", "Noto Sans KR"],
       "zh_CN": ["Microsoft YaHei UI", "Noto Sans CJK SC", "Noto Sans SC", "PingFang SC"]}


def font(size: float = 13, weight: int = 400, family: str = SANS) -> QFont:
    from openkick.i18n import language
    f = QFont(family)
    if language() in CJK:
        f.setFamilies([family] + CJK[language()])
    f.setPixelSize(round(size))
    f.setWeight(QFont.Weight(weight))
    f.setHintingPreference(QFont.PreferNoHinting)
    return f


def mono(size: float = 13, weight: int = 400) -> QFont:
    return font(size, weight, MONO)


def display(size: float = 24, weight: int = 700) -> QFont:
    f = font(size, weight, DISPLAY)
    f.setLetterSpacing(QFont.PercentageSpacing, 97)
    return f


def menu_title(menu, text: str):
    """A visible heading inside a menu. Qt's addSection() text is not drawn with our style
    sheet, so groups of entries would look all alike."""
    from PySide6.QtWidgets import QLabel, QWidgetAction
    if not menu.isEmpty():
        menu.addSeparator()
    label = QLabel(text.upper())
    label.setStyleSheet(f"color: {FAINT}; font-size: 10px; font-weight: 600; letter-spacing: 1px; padding: 6px 12px 3px 12px;"
                        " background: transparent;")
    action = QWidgetAction(menu)
    action.setDefaultWidget(label)
    action.setEnabled(False)
    menu.addAction(action)
    return action


def qcolor(hex_color: str, alpha: int | None = None) -> QColor:
    c = QColor(hex_color)
    if alpha is not None:
        c.setAlpha(alpha)
    return c


def text_on(hex_color: str) -> str:
    """Dark or light text for a colored chip (contrast by luminance)."""
    c = QColor(hex_color)
    lum = 0.2126 * c.redF() ** 2.2 + 0.7152 * c.greenF() ** 2.2 + 0.0722 * c.blueF() ** 2.2
    return "#14110A" if lum > 0.18 else "#F4F6F8"


STYLE = f"""
QWidget {{ color: {TEXT}; }}
QMainWindow, QDialog, QStackedWidget, #Page {{ background: {BG}; }}
QWidget#Panel, QFrame#Panel {{ background: {PANEL}; }}
QToolTip {{ background: {RAISED}; color: {TEXT}; border: 1px solid {BORDER}; border-radius: 6px; padding: 6px 8px; }}

QMenuBar {{ background: transparent; border: 0; }}
QMenuBar::item {{ background: transparent; color: {MUTED}; padding: 7px 10px; border-radius: 6px; }}
QMenuBar::item:selected, QMenuBar::item:pressed {{ background: {ACTIVE}; color: {TEXT}; }}
QMenu {{ background: {RAISED}; border: 1px solid {BORDER}; border-radius: 10px; padding: 6px; }}
QMenu::item {{ padding: 7px 28px 7px 12px; border-radius: 6px; color: {TEXT}; }}
QMenu::item:selected {{ background: {ACTIVE}; }}
QMenu::item:disabled {{ color: {DIM}; }}
QMenu::separator {{ height: 1px; background: {LINE}; margin: 5px 8px; }}
QMenu::icon {{ padding-left: 8px; }}
QMenu::indicator {{ width: 14px; height: 14px; margin-left: 6px; }}

QPushButton {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px; padding: 0 14px;
              min-height: 34px; font-weight: 500; }}
QPushButton:hover {{ background: #1C212A; border-color: #3A4352; }}
QPushButton:pressed {{ background: {ACTIVE}; }}
QPushButton:disabled {{ color: {DIM}; border-color: {LINE}; }}
QPushButton[primary="true"] {{ background: {ACCENT}; color: {ON_ACCENT}; border: 0; font-weight: 600; }}
QPushButton[primary="true"]:hover {{ background: {ACCENT_HOVER}; }}
QPushButton[primary="true"]:disabled {{ background: #4B5A24; color: #1F2410; }}
QPushButton[ghost="true"] {{ background: transparent; border: 0; color: {ACCENT}; padding: 0 8px; }}
QPushButton[ghost="true"]:hover {{ background: {RAISED}; }}
QPushButton[warn="true"] {{ background: #2A2016; border: 0; color: {WARN_TEXT}; min-height: 26px; padding: 0 10px; }}
QPushButton::menu-indicator {{ image: none; width: 0; }}

QToolButton {{ background: transparent; border: 0; border-radius: 8px; padding: 4px; }}
QToolButton:hover {{ background: {RAISED}; }}
QToolButton:checked {{ background: {ACTIVE}; }}
QToolButton#RailButton {{ border-radius: 10px; }}
QToolButton#RailButton:checked {{ background: {ACTIVE}; border: 1px solid #3A4352; }}
QToolButton::menu-indicator {{ image: none; width: 0; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QDateEdit, QComboBox, QPlainTextEdit, QTextEdit {{
    background: {CARD_2}; border: 1px solid {BORDER}; border-radius: 8px; padding: 6px 10px;
    selection-background-color: #3B4A1A; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QDateEdit:focus, QComboBox:focus, QPlainTextEdit:focus {{ border-color: {ACCENT}; }}
QDateEdit:disabled {{ color: {DIM}; }}
QDateEdit::drop-down {{ border: 0; width: 22px; }}
QComboBox {{ min-height: 20px; }}
QComboBox::drop-down {{ border: 0; width: 22px; }}
QComboBox QAbstractItemView {{ background: {RAISED}; border: 1px solid {BORDER}; selection-background-color: {ACTIVE}; outline: 0; padding: 4px; }}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 0; border: 0; }}

QCheckBox, QRadioButton {{ spacing: 8px; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 16px; height: 16px; }}
QCheckBox::indicator {{ border: 1.5px solid {BORDER}; border-radius: 5px; background: transparent; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; image: url("{CHECK_SVG}"); }}
QListView::indicator {{ width: 16px; height: 16px; border: 1.5px solid {BORDER}; border-radius: 5px; background: transparent; }}
QListView::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; image: url("{CHECK_SVG}"); }}
QMenu::indicator:checked {{ image: url("{CHECK_SVG}"); background: {ACCENT}; border-radius: 4px; }}
QRadioButton::indicator {{ border: 2px solid {BORDER}; border-radius: 8px; background: transparent; }}
QRadioButton::indicator:checked {{ border-color: {ACCENT}; background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5, stop:0 {ACCENT}, stop:0.45 {ACCENT}, stop:0.5 transparent); }}

QTabWidget::pane {{ border: 0; }}
QTabBar {{ background: transparent; }}
QTabBar::tab {{ background: transparent; color: {MUTED}; padding: 10px 12px 9px; border: 0; border-bottom: 2px solid transparent; margin-right: 4px; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom-color: {ACCENT}; font-weight: 600; }}
QTabBar::tab:hover:!selected {{ color: {TEXT_2}; }}

QScrollArea {{ border: 0; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #2A303A; border-radius: 3px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #3A4352; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #2A303A; border-radius: 3px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QProgressBar {{ background: #222834; border: 0; border-radius: 4px; height: 8px; text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 4px; }}

QTreeView, QListView {{ background: {CARD_2}; border: 1px solid {LINE}; border-radius: 8px;
    selection-background-color: {ACTIVE}; selection-color: {TEXT}; outline: 0; }}
QTreeView::item, QListView::item {{ padding: 3px 2px; }}
QTreeView::item:hover, QListView::item:hover {{ background: {RAISED}; }}
QFileDialog, QColorDialog, QFontDialog {{ background: {PANEL}; }}
QFileDialog QToolButton {{ border: 1px solid {BORDER}; background: {CARD}; }}
QTableWidget, QTableView {{ background: {PANEL}; border: 1px solid #1E232C; border-radius: 12px; gridline-color: {LINE_SOFT};
    alternate-background-color: #11151B; selection-background-color: {ACTIVE}; }}
QHeaderView::section {{ background: #131820; color: {MUTED}; border: 0; border-bottom: 1px solid {LINE_SOFT}; padding: 9px 12px; font-weight: 500; }}
QTableCornerButton::section {{ background: #131820; border: 0; }}

QSlider::groove:horizontal {{ height: 4px; background: #222834; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 14px; height: 14px; margin: -5px 0; border-radius: 7px; }}

QLabel#SectionTitle {{ color: {MUTED}; font-size: 11px; font-weight: 600; letter-spacing: 1px; }}
QLabel#Muted {{ color: {MUTED}; }}
QLabel#Faint {{ color: {FAINT}; font-size: 12px; }}
QFrame#Card {{ background: {CARD}; border: 1px solid #222834; border-radius: 10px; }}
QFrame#CardSelected {{ background: #171D12; border: 1px solid {ACCENT}; border-radius: 10px; }}
QFrame#Separator {{ background: {LINE}; max-height: 1px; min-height: 1px; }}
QStatusBar {{ background: {PANEL}; color: {MUTED}; border-top: 1px solid {LINE}; }}
QMessageBox {{ background: {RAISED}; }}
QMessageBox QLabel {{ color: {TEXT}; }}
QInputDialog {{ background: {RAISED}; }}
"""
