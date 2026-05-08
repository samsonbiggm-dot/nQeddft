"""
QED-DFT Studio v1.3
四个 Tab：⚛ 结构 / ⚙ 计算设置 / 📋 任务监控 / 📊 结果
nQEDDFT 参数全部整合到「计算设置」，通过连续任务流水线驱动。
"""
from __future__ import annotations
import sys, os, json, logging, time
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QPushButton, QPlainTextEdit,
    QTableWidget, QTableWidgetItem, QProgressBar,
    QGroupBox, QSplitter, QFileDialog, QScrollArea,
    QMessageBox, QStatusBar, QFrame, QHeaderView,
    QDialog, QDialogButtonBox, QSizePolicy, QStyledItemDelegate,
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QFont, QColor, QAction

# ── matplotlib ────────────────────────────────────────────
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    import warnings
    _CJK = ["Microsoft YaHei","SimHei","PingFang SC",
             "Noto Sans CJK SC","WenQuanYi Micro Hei","Arial Unicode MS"]
    _avail = {f.name for f in mpl.font_manager.fontManager.ttflist}
    _cjk   = next((f for f in _CJK if f in _avail), None)
    if _cjk: mpl.rcParams["font.sans-serif"] = [_cjk] + mpl.rcParams["font.sans-serif"]
    mpl.rcParams["axes.unicode_minus"] = False
    warnings.filterwarnings("ignore", message="Glyph .* missing from font",
                             category=UserWarning, module="matplotlib")
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import gemmi;            HAS_GEMMI   = True
except ImportError:          HAS_GEMMI   = False
try:
    from ase.io import read as ase_read; HAS_ASE = True
except ImportError:          HAS_ASE     = False
try:
    from nqeddft import Cavity, QEDRKS, QEDUKS; HAS_NQEDDFT = True
except ImportError:          HAS_NQEDDFT = False

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════
#  内嵌物化数据
# ═══════════════════════════════════════════════════════════

AU2EV = 27.2114

def cm_to_au(nu: float) -> float:
    return nu / 219474.6

CLUSTERS = {
    "Cu1": {"geom": "Cu   0.000000   0.000000   0.000000",
             "top_xyz": (0.,0.,0.), "spin": 1, "method": "QEDUKS"},
    "Cu2": {"geom": "Cu   0.000000   0.000000   0.000000\nCu   2.556191   0.000000   0.000000",
             "top_xyz": (0.,0.,0.), "spin": 0, "method": "QEDRKS"},
    "Cu4": {"geom": ("Cu   1.475818   0.000000   0.000000\n"
                     "Cu  -0.737909   1.278096   0.000000\n"
                     "Cu  -0.737909  -1.278096   0.000000\n"
                     "Cu   0.000000   0.000000   2.087121"),
             "top_xyz": (0.,0.,2.087121), "spin": 0, "method": "QEDRKS"},
    "Cu8": {"geom": ("Cu   0.000000   0.000000   0.000000\n"
                     "Cu   2.556191   0.000000   0.000000\n"
                     "Cu   1.278096   2.213726   0.000000\n"
                     "Cu   3.834287   2.213726   0.000000\n"
                     "Cu   1.278096   0.737909   2.087121\n"
                     "Cu   3.834287   0.737909   2.087121\n"
                     "Cu   2.556191   2.951635   2.087121\n"
                     "Cu   5.112382   2.951635   2.087121"),
             "top_xyz": (1.278096,0.737909,2.087121), "spin": 0, "method": "QEDRKS"},
}

ADS_OFFSETS = {
    "CO2*": {
        "geom_offset_Cu1": ("C    0.000000   0.000000   1.850000\n"
                             "O    1.160074   0.000000   2.390951\n"
                             "O   -1.087569   0.000000   2.357142"),
        "geom_offset":     ("C    0.000000   0.000000   2.000000\n"
                             "O    1.160074   0.000000   2.540951\n"
                             "O   -1.087569   0.000000   2.507142"),
        "n_electrons": 16, "nu_target_cm": 1720,
    },
    "COOH*": {
        "geom_offset_Cu1": ("C    0.000000   0.000000   2.000000\n"
                             "O    1.028938   0.000000   2.655506\n"
                             "O   -1.130145   0.000000   2.719981\n"
                             "H   -2.041646   0.000000   2.388222"),
        "geom_offset":     ("C    0.000000   0.000000   2.000000\n"
                             "O    1.028938   0.000000   2.655506\n"
                             "O   -1.130145   0.000000   2.719981\n"
                             "H   -2.041646   0.000000   2.388222"),
        "n_electrons": 17, "nu_target_cm": 1640,
    },
    "CO*": {
        "geom_offset_Cu1": ("C    0.000000   0.000000   1.950000\n"
                             "O    0.000000   0.000000   3.100000"),
        "geom_offset":     ("C    0.000000   0.000000   1.950000\n"
                             "O    0.000000   0.000000   3.100000"),
        "n_electrons": 10, "nu_target_cm": 2050,
    },
}

GAS_REFS = {
    "CO2":  ("C   0.000000   0.000000   0.000000\n"
              "O   1.162000   0.000000   0.000000\n"
              "O  -1.162000   0.000000   0.000000"),
    "CO":   ("C   0.000000   0.000000   0.000000\n"
              "O   0.000000   0.000000   1.128000"),
    "COOH": ("C   0.000  0.000  0.000\n"
              "O   1.250  0.000  0.000\n"
              "O  -0.550  1.140  0.000\n"
              "H  -1.530  1.100  0.000"),
    "H2":   "H   0.0  0.0  0.0\nH   0.0  0.0  0.741",
    "H2O":  "O   0.0  0.0  0.0\nH   0.757  0.0  0.586\nH  -0.757  0.0  0.586",
}

LAMBDA_PRESETS = [0.0, 0.005, 0.01, 0.02, 0.05]
BASIS_NQED = {"Cu": "def2-SVP", "default": "cc-pVDZ"}
GAS_MAP = {"CO2*": "CO2", "COOH*": "COOH", "CO*": "CO"}
COMMON_ELEMENTS = [
    "H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I",
    "Li", "Na", "K", "Mg", "Ca", "Al", "Si",
    "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ag", "Au", "Pt", "Pd",
]

# ═══════════════════════════════════════════════════════════
#  国际化 / Internationalisation
# ═══════════════════════════════════════════════════════════

LANG = "zh"   # 全局语言状态，"zh" | "en"

TR = {
    # ── Window / Tabs ───────────────────────────────────────
    "app_title":         {"zh": "QED-DFT Studio  v1.3",    "en": "QED-DFT Studio  v1.3"},
    "tab_struct":        {"zh": "⚛  结构",                  "en": "⚛  Structure"},
    "tab_settings":      {"zh": "⚙  计算设置",              "en": "⚙  Settings"},
    "tab_monitor":       {"zh": "📋  任务监控",              "en": "📋  Monitor"},
    "tab_results":       {"zh": "📊  结果",                  "en": "📊  Results"},
    # ── Menu ────────────────────────────────────────────────
    "menu_file":         {"zh": "文件 (&F)",                 "en": "File (&F)"},
    "menu_new":          {"zh": "新建项目",                  "en": "New Project"},
    "menu_open":         {"zh": "打开项目",                  "en": "Open Project"},
    "menu_save":         {"zh": "保存项目",                  "en": "Save Project"},
    "menu_export":       {"zh": "导出结果...",               "en": "Export Results..."},
    "menu_exit":         {"zh": "退出",                      "en": "Exit"},
    "menu_calc":         {"zh": "计算 (&C)",                 "en": "Calc (&C)"},
    "menu_submit":       {"zh": "▶ 提交计算",                "en": "▶ Submit"},
    "menu_abort":        {"zh": "◼ 中止计算",                "en": "◼ Abort"},
    "menu_cmd":          {"zh": "编辑提交命令...",            "en": "Edit Command..."},
    "menu_view":         {"zh": "视图 (&V)",                 "en": "View (&V)"},
    "menu_lang_zh":      {"zh": "中文",                      "en": "中文 (Chinese)"},
    "menu_lang_en":      {"zh": "English",                   "en": "English"},
    "menu_help":         {"zh": "帮助 (&H)",                 "en": "Help (&H)"},
    "menu_about":        {"zh": "关于",                      "en": "About"},
    # ── Toolbar ─────────────────────────────────────────────
    "tb_submit":         {"zh": "▶  提交计算",               "en": "▶  Submit"},
    "tb_abort":          {"zh": "◼  中止",                   "en": "◼  Abort"},
    "tb_save":           {"zh": "💾  保存项目",              "en": "💾  Save"},
    "tb_cmd":            {"zh": "⚙  提交命令",               "en": "⚙  Command"},
    # ── Structure Panel ──────────────────────────────────────
    "btn_h2o":           {"zh": "示例 H₂O",                  "en": "H₂O Sample"},
    "btn_h2":            {"zh": "示例 H₂",                   "en": "H₂ Sample"},
    "btn_load_xyz":      {"zh": "导入 .xyz",                  "en": "Open .xyz"},
    "btn_load_cif":      {"zh": "导入 .cif",                  "en": "Open .cif"},
    "btn_save_xyz":      {"zh": "💾 保存 .xyz",               "en": "💾 Save .xyz"},
    "btn_3d":            {"zh": "🔭 3D 预览",                 "en": "🔭 3D View"},
    "btn_add":           {"zh": "+ 原子",                     "en": "+ Atom"},
    "btn_del":           {"zh": "− 删除",                     "en": "− Delete"},
    "btn_place_ads":     {"zh": "放置吸附物",                  "en": "Place Adsorbate"},
    "btn_apply_xyz":     {"zh": "✓ 应用文本",                 "en": "✓ Apply Text"},
    "col_elem":          {"zh": "元素",                       "en": "Element"},
    "col_x":             {"zh": "X (Å)",                      "en": "X (Å)"},
    "col_y":             {"zh": "Y (Å)",                      "en": "Y (Å)"},
    "col_z":             {"zh": "Z (Å)",                      "en": "Z (Å)"},
    "tab_table":         {"zh": "🔢 坐标表格",                "en": "🔢 Table"},
    "tab_xyztext":       {"zh": "📝 XYZ 文本",                "en": "📝 XYZ Text"},
    "tab_3d_view":       {"zh": "🔭 3D 视图",                 "en": "🔭 3D View"},
    "lbl_struct_hint":   {"zh": "在表格中直接编辑原子坐标，或切换到「XYZ 文本」选项卡直接编写/粘贴 .xyz 格式内容",
                          "en": "Edit atoms directly in the table, or switch to XYZ Text tab to paste/edit raw .xyz content"},
    "xyz_placeholder":   {"zh": "在此处粘贴或编辑 .xyz 格式内容，例如：\n\n3\nwater molecule\nO   0.000   0.000   0.119\nH   0.757   0.000  -0.477\nH  -0.757   0.000  -0.477\n\n点击「✓ 应用文本」以解析并同步到坐标表格。",
                          "en": "Paste or edit .xyz content here, e.g.:\n\n3\nwater molecule\nO   0.000   0.000   0.119\nH   0.757   0.000  -0.477\nH  -0.757   0.000  -0.477\n\nClick '✓ Apply Text' to parse and sync to the table."},
    "err_xyz_parse":     {"zh": "XYZ 解析错误",              "en": "XYZ Parse Error"},
    "err_xyz_detail":    {"zh": "XYZ 解析错误:",             "en": "XYZ parse error:"},
    "ok_xyz_live":       {"zh": "✓ XYZ 有效: {n} atoms",      "en": "✓ Valid XYZ: {n} atoms"},
    "err_xyz_live":      {"zh": "✗ 第 {line} 行: {msg}",      "en": "✗ line {line}: {msg}"},
    "err_table_live":    {"zh": "表格中有非法元素或坐标,已标红。", "en": "Invalid element or coordinate in table; highlighted."},
    "err_cif_fail":      {"zh": "失败",                      "en": "Failed"},
    "err_cif_install":   {"zh": "请安装 gemmi 或 ase",       "en": "Please install gemmi or ase"},
    "dlg_3d_title":      {"zh": "3D 预览",                   "en": "3D Preview"},
    "dlg_3d_close":      {"zh": "关闭",                      "en": "Close"},
    # ── Calc Settings ────────────────────────────────────────
    "grp_method":        {"zh": "计算方法",                  "en": "Method"},
    "lbl_method":        {"zh": "方法:",                     "en": "Method:"},
    "lbl_xc":            {"zh": "XC 泛函:",                  "en": "XC Functional:"},
    "lbl_basis":         {"zh": "基组:",                     "en": "Basis Set:"},
    "lbl_charge":        {"zh": "电荷:",                     "en": "Charge:"},
    "lbl_mult":          {"zh": "多重度:",                   "en": "Multiplicity:"},
    "lbl_auto_spin":     {"zh": "nQEDDFT 模式下自动推算自旋", "en": "Spin auto-derived in nQEDDFT mode"},
    "grp_scf":           {"zh": "SCF 参数",                  "en": "SCF Parameters"},
    "grp_advanced":      {"zh": "高级设置",                  "en": "Advanced Settings"},
    "advanced_hint":     {"zh": "SCF、断点续算和子进程流水线等低频参数收纳在这里。",
                          "en": "SCF, checkpoint, and subprocess pipeline options live here."},
    "lbl_maxcyc":        {"zh": "最大迭代:",                 "en": "Max Cycles:"},
    "lbl_tol":           {"zh": "收敛阈值 (Ha):",            "en": "Conv. Threshold (Ha):"},
    "lbl_diis":          {"zh": "DIIS 空间:",                "en": "DIIS Space:"},
    "lbl_ls":            {"zh": "Level shift:",              "en": "Level Shift:"},
    "grp_sys":           {"zh": "nQEDDFT 体系构型",          "en": "nQEDDFT System"},
    "lbl_cluster":       {"zh": "Cu 团簇:",                  "en": "Cu Cluster:"},
    "lbl_ads":           {"zh": "吸附物:",                   "en": "Adsorbate:"},
    "ads_none":          {"zh": "无（裸团簇）",              "en": "None (bare cluster)"},
    "lbl_gasref":        {"zh": "气相参考:",                 "en": "Gas Reference:"},
    "gasref_auto":       {"zh": "自动推断",                  "en": "Auto"},
    "btn_sync_geom":     {"zh": "📌 同步几何到结构面板",     "en": "📌 Sync Geometry to Structure Tab"},
    "lbl_cluster_atoms": {"zh": "{n} Cu 原子 · {st}",       "en": "{n} Cu atoms · {st}"},
    "lbl_ads_info":      {"zh": "价电子 {ne} · 目标频率 {nu} cm⁻¹", "en": "valence e⁻ {ne} · target ν {nu} cm⁻¹"},
    "lbl_preview_cluster_title": {"zh": "团簇结构预览",      "en": "Cluster Preview"},
    "lbl_preview_scan_title":    {"zh": "λ 扫描预览（计算中实时更新）", "en": "λ Scan Preview (live update)"},
    "lbl_custom_lambda": {"zh": "自定义序列:",               "en": "Custom sequence:"},
    "lbl_checkpoint":    {"zh": "检查点文件:",               "en": "Checkpoint file:"},
    "grp_cav":           {"zh": "腔场参数 (Cavity)",         "en": "Cavity Parameters"},
    "lbl_omega":         {"zh": "ω_c (a.u.):",              "en": "ω_c (a.u.):"},
    "lbl_lambda":        {"zh": "λ 耦合强度:",              "en": "λ Coupling:"},
    "lbl_pol":           {"zh": "极化方向:",                 "en": "Polarization:"},
    "pol_auto":          {"zh": "自动（优化构型提取）",       "en": "Auto (from optimised geometry)"},
    "pol_z":             {"zh": "z 轴 [0,0,1]",             "en": "z-axis [0,0,1]"},
    "pol_x":             {"zh": "x 轴 [1,0,0]",             "en": "x-axis [1,0,0]"},
    "pol_y":             {"zh": "y 轴 [0,1,0]",             "en": "y-axis [0,1,0]"},
    "grp_pipe":          {"zh": "计算任务流水线（按序执行）", "en": "Pipeline (sequential)"},
    "pipe_hint":         {"zh": "勾选的任务将依次自动执行，未勾选的跳过。",
                          "en": "Checked tasks run in order; unchecked tasks are skipped."},
    "chk_geom":          {"zh": "① 几何优化（λ=0 参考构型）","en": "① Geometry optimisation (λ=0 ref.)"},
    "chk_gas":           {"zh": "② 气相参考能量计算",        "en": "② Gas-phase reference energy"},
    "chk_freq0":         {"zh": "③ 无腔振动分析 → ν₀(C-O)", "en": "③ Cavity-free vibration → ν₀(C-O)"},
    "chk_scan":          {"zh": "④ λ 扫描（多耦合强度 SCF）","en": "④ λ scan (multi-coupling SCF)"},
    "chk_freqcav":       {"zh": "⑤ 腔中振动分析 → Δω 频移", "en": "⑤ In-cavity vibration → Δω shift"},
    "chk_pol":           {"zh": "⑥ 极化激元分析",           "en": "⑥ Polariton analysis"},
    "grp_scan_sub":      {"zh": "λ 扫描子参数",             "en": "λ Scan Parameters"},
    "chk_preset":        {"zh": "使用预设序列 [0, 0.005, 0.01, 0.02, 0.05]",
                          "en": "Use preset sequence [0, 0.005, 0.01, 0.02, 0.05]"},
    "lbl_custom_seq":    {"zh": "自定义序列:",               "en": "Custom sequence:"},
    "grp_ckpt":          {"zh": "断点续算",                  "en": "Checkpoint / Resume"},
    "lbl_ckpt":          {"zh": "检查点文件:",               "en": "Checkpoint file:"},
    "chk_resume":        {"zh": "恢复已有检查点（跳过已完成步骤）",
                          "en": "Resume from checkpoint (skip completed steps)"},
    "lbl_preview_cluster": {"zh": "团簇结构预览",            "en": "Cluster Preview"},
    "lbl_preview_scan":  {"zh": "λ 扫描预览（计算中实时更新）","en": "λ Scan Preview (live)"},
    # ── Monitor ──────────────────────────────────────────────
    "grp_pipeline_status": {"zh": "任务流水线状态",          "en": "Pipeline Status"},
    "grp_task_queue":    {"zh": "任务队列",                  "en": "Task Queue"},
    "col_task":          {"zh": "任务",                      "en": "Task"},
    "col_state":         {"zh": "状态",                      "en": "State"},
    "col_started":       {"zh": "开始时间",                  "en": "Started"},
    "col_elapsed":       {"zh": "耗时",                      "en": "Elapsed"},
    "col_workdir":       {"zh": "工作目录",                  "en": "Workdir"},
    "state_idle":        {"zh": "等待",                      "en": "Idle"},
    "state_running":     {"zh": "运行中",                    "en": "Running"},
    "state_done":        {"zh": "完成",                      "en": "Done"},
    "state_skip":        {"zh": "跳过",                      "en": "Skipped"},
    "state_error":       {"zh": "错误",                      "en": "Error"},
    "step_geom":         {"zh": "① 几何优化",               "en": "① Geom Opt"},
    "step_gas":          {"zh": "② 气相参考",               "en": "② Gas Ref"},
    "step_freq0":        {"zh": "③ 无腔振动",               "en": "③ Free Vib"},
    "step_scan":         {"zh": "④ λ 扫描",                 "en": "④ λ Scan"},
    "step_freqcav":      {"zh": "⑤ 腔中振动",               "en": "⑤ Cav Vib"},
    "step_pol":          {"zh": "⑥ 极化激元",               "en": "⑥ Polariton"},
    "status_ready":      {"zh": "就绪",                      "en": "Ready"},
    "flow_struct":       {"zh": "结构准备",                  "en": "Structure"},
    "flow_settings":     {"zh": "参数设置",                  "en": "Settings"},
    "flow_monitor":      {"zh": "任务运行",                  "en": "Run"},
    "flow_results":      {"zh": "结果分析",                  "en": "Results"},
    # ── Results ──────────────────────────────────────────────
    "card_energy":       {"zh": "总能量",                    "en": "Total Energy"},
    "card_ads":          {"zh": "吸附能",                    "en": "Adsorption Energy"},
    "card_freq0":        {"zh": "ν₀ C-O",                   "en": "ν₀ C-O"},
    "card_dfreq":        {"zh": "Δω (腔)",                  "en": "Δω (Cavity)"},
    "card_conv":         {"zh": "收敛",                      "en": "Converged"},
    "tab_vib":           {"zh": "振动频率",                  "en": "Vibration Freq."},
    "tab_scan_res":      {"zh": "λ 扫描结果",               "en": "λ Scan Results"},
    "tab_polariton":     {"zh": "极化激元",                  "en": "Polariton"},
    "tab_summary":       {"zh": "数值汇总",                  "en": "Summary"},
    "btn_save_res":      {"zh": "💾 保存结果",               "en": "💾 Save Results"},
    "tab_lam_plot":      {"zh": "λ 扫描图",                 "en": "λ Scan Plot"},
    "tab_shift_plot":    {"zh": "频移对比",                  "en": "Freq. Shift"},
    "col_mode":          {"zh": "模式",                      "en": "Mode"},
    "col_freq":          {"zh": "频率 (cm⁻¹)",              "en": "Freq. (cm⁻¹)"},
    "col_intens":        {"zh": "强度",                      "en": "Intensity"},
    "col_annot":         {"zh": "标注",                      "en": "Label"},
    "col_lam":           {"zh": "λ (a.u.)",                 "en": "λ (a.u.)"},
    "col_e_ha":          {"zh": "E (Ha)",                   "en": "E (Ha)"},
    "col_de_ads":        {"zh": "ΔE_ads (eV)",             "en": "ΔE_ads (eV)"},
    "col_co_freq":       {"zh": "ν C-O (cm⁻¹)",            "en": "ν C-O (cm⁻¹)"},
    "col_dw":            {"zh": "Δω (cm⁻¹)",               "en": "Δω (cm⁻¹)"},
    "col_conv":          {"zh": "收敛",                      "en": "Conv."},
    "col_branch":        {"zh": "支路",                      "en": "Branch"},
    "col_phot":          {"zh": "光子权重",                  "en": "Photon wt."},
    "col_mat":           {"zh": "物质权重",                  "en": "Matter wt."},
    "col_param":         {"zh": "参数",                      "en": "Parameter"},
    "col_value":         {"zh": "数值",                      "en": "Value"},
    "col_unit":          {"zh": "单位",                      "en": "Unit"},
    "col_note":          {"zh": "备注",                      "en": "Note"},
    # ── Statusbar / Misc ─────────────────────────────────────
    "status_nqed_ok":    {"zh": "nqeddft ✓",               "en": "nqeddft ✓"},
    "status_nqed_miss":  {"zh": "nqeddft 未安装",           "en": "nqeddft not installed"},
    "warn_no_atoms":     {"zh": "请先在「结构」选项卡输入原子坐标，\n或在「计算设置」中启用 nQEDDFT 团簇模式。",
                          "en": "Please enter atom coordinates in the Structure tab,\nor enable nQEDDFT cluster mode in Settings."},
    "warn_preflight":    {"zh": "提交前检查未通过:",          "en": "Preflight check failed:"},
    "warn_bad_lambda":   {"zh": "λ 序列为空或包含非法数值。", "en": "The lambda sequence is empty or contains invalid values."},
    "warn_bad_workdir":  {"zh": "工作目录不可写:",            "en": "Workdir is not writable:"},
    "warn_stage_script": {"zh": "找不到 Stage 脚本:",         "en": "Stage script not found:"},
    "task_manifest_saved": {"zh": "任务配置已写入:",          "en": "Task manifest saved:"},
    "warn_title":        {"zh": "警告",                     "en": "Warning"},
    "err_title":         {"zh": "计算错误",                 "en": "Calculation Error"},
    "save_ok":           {"zh": "保存成功",                  "en": "Saved"},
    "save_ok_msg":       {"zh": "已保存:",                   "en": "Saved to:"},
    "proj_saved":        {"zh": "已保存:",                   "en": "Saved:"},
    "cmd_updated":       {"zh": "命令已更新:",               "en": "Command updated:"},
    "about_title":       {"zh": "关于 QED-DFT Studio",      "en": "About QED-DFT Studio"},
    "dlg_save_proj":     {"zh": "保存项目",                  "en": "Save Project"},
    "dlg_save_res":      {"zh": "保存结果",                  "en": "Save Results"},
    "dlg_save_xyz":      {"zh": "保存 XYZ 文件",             "en": "Save XYZ File"},
    # Results panel
    "conv_yes":          {"zh": "✓ 已收敛",                  "en": "✓ Converged"},
    "conv_no":           {"zh": "✗ 未收敛",                  "en": "✗ Not converged"},
    "sum_total_e":       {"zh": "总能量",                    "en": "Total Energy"},
    "sum_ads_e":         {"zh": "吸附能 ΔE_ads",            "en": "Adsorption Energy ΔE_ads"},
    "sum_freq0":         {"zh": "ν₀ C-O（无腔）",           "en": "ν₀ C-O (free)"},
    "sum_freq_cav":      {"zh": "ν C-O（腔中）",            "en": "ν C-O (cavity)"},
    "sum_shift":         {"zh": "腔诱导频移 Δω",            "en": "Cavity-induced shift Δω"},
    "sum_dipole":        {"zh": "偶极矩 |μ|",               "en": "Dipole moment |μ|"},
    "sum_walltime":      {"zh": "计算耗时",                  "en": "Wall time"},
    "sum_scf_ref":       {"zh": "SCF 收敛值",               "en": "SCF converged"},
    "sum_ads_ref":       {"zh": "相对气相参考",              "en": "rel. to gas ref."},
    "sum_vib_ref":       {"zh": "最强伸缩振动",              "en": "strongest stretch"},
    "sum_cav_ref":       {"zh": "λ 耦合后",                 "en": "after λ coupling"},
    "sum_shift_ref":     {"zh": "+蓝移/-红移",              "en": "+blue/-red shift"},
    "sum_pipe_ref":      {"zh": "流水线总时间",              "en": "pipeline total"},
    # Plot titles (matplotlib)
    "plt_ads_vs_lam":    {"zh": "吸附能 vs λ",              "en": "Adsorption energy vs λ"},
    "plt_freq_vs_lam":   {"zh": "C-O 频率 vs λ",            "en": "C-O frequency vs λ"},
    "plt_shift_title":   {"zh": "腔诱导 C-O 频移",          "en": "Cavity-induced C-O shift"},
    "plt_3d_placeholder":{"zh": "导入结构文件\n以显示 3D 视图","en": "Import a structure\nto show 3D view"},
    "plt_n_atoms":       {"zh": "{n} 原子",                  "en": "{n} atoms"},
    # Command dialog
    "cmd_dlg_title":     {"zh": "编辑提交命令",              "en": "Edit Submit Command"},
    "cmd_lbl_template":  {"zh": "模板:",                     "en": "Template:"},
    "cmd_col_var":       {"zh": "变量",                      "en": "Variable"},
    "cmd_col_val":       {"zh": "值",                        "en": "Value"},
    "cmd_local":         {"zh": "本地运行",                  "en": "Local"},
    "cmd_custom":        {"zh": "自定义",                    "en": "Custom"},
    # About dialog
    "about_nqed_ok":     {"zh": "✓ 已安装",                 "en": "✓ Installed"},
    "about_nqed_miss":   {"zh": "✗ 未安装 (pip install nqeddft)", "en": "✗ Not installed (pip install nqeddft)"},
    "about_body":        {"zh": ("<h2>QED-DFT Studio v1.3</h2>"
                                  "<p>量子电动力学密度泛函理论计算平台</p>"
                                  "<p>PySCF · nQEDDFT · PySide6</p><hr/>"
                                  "<b>界面架构（四 Tab）：</b><br/>"
                                  "⚛ 结构 → ⚙ 计算设置 → 📋 任务监控 → 📊 结果<br/><br/>"
                                  "<b>「计算设置」整合内容：</b><br/>"
                                  "• 通用参数（方法 / 基组 / SCF / 多重度）<br/>"
                                  "• nQEDDFT 团簇 + 吸附物（内置 Cu₁~Cu₈ / CO₂*/COOH*/CO*）<br/>"
                                  "• 腔场参数（ω_c / λ / 极化方向）<br/>"
                                  "• 连续任务流水线：几何优化→气相参考→无腔振动→λ扫描→腔中振动→极化激元<br/>"
                                  "• 断点续算（JSON 检查点）<br/>"
                                  "• λ 扫描实时预览图（右侧同步更新）<br/><br/>"),
                           "en": ("<h2>QED-DFT Studio v1.3</h2>"
                                  "<p>Quantum Electrodynamical Density Functional Theory Platform</p>"
                                  "<p>PySCF · nQEDDFT · PySide6</p><hr/>"
                                  "<b>Layout (4 tabs):</b><br/>"
                                  "⚛ Structure → ⚙ Settings → 📋 Monitor → 📊 Results<br/><br/>"
                                  "<b>Settings panel includes:</b><br/>"
                                  "• General parameters (method / basis / SCF / multiplicity)<br/>"
                                  "• nQEDDFT cluster + adsorbate (built-in Cu₁~Cu₈ / CO₂*/COOH*/CO*)<br/>"
                                  "• Cavity parameters (ω_c / λ / polarisation)<br/>"
                                  "• Pipeline: geom opt → gas ref → free vib → λ scan → cav vib → polariton<br/>"
                                  "• Checkpoint / resume<br/>"
                                  "• Live λ scan preview (right panel)<br/><br/>")},
    # Step labels emitted from Worker (shown in status bar)
    "worker_geom":       {"zh": "① 几何优化...",            "en": "① Geometry optimisation..."},
    "worker_gas":        {"zh": "② 气相参考 {gk}...",       "en": "② Gas reference {gk}..."},
    "worker_freq0":      {"zh": "③ 无腔振动分析...",         "en": "③ Free vibration analysis..."},
    "worker_scan":       {"zh": "④ λ 扫描 {i}/{n}  λ={lam:.4f}...", "en": "④ λ scan {i}/{n}  λ={lam:.4f}..."},
    "worker_freqcav":    {"zh": "⑤ 腔中振动分析...",         "en": "⑤ Cavity vibration analysis..."},
    "worker_pol":        {"zh": "⑥ 极化激元分析...",         "en": "⑥ Polariton analysis..."},
    "worker_done":       {"zh": "✓ 计算完成",               "en": "✓ Calculation complete"},
    "status_ready_full": {"zh": "就绪",                      "en": "Ready"},
    "3d_n_atoms":        {"zh": "{n} 原子",                  "en": "{n} atoms"},
    "struct_placeholder": {"zh": "导入结构文件\n以显示 3D 视图", "en": "Import a structure file\nto display 3D view"},
    "log_placeholder":   {"zh": "计算日志将在此处实时显示...\n请在「计算设置」中配置参数后点击「▶ 提交计算」。",
                          "en": "Calculation log appears here...\nConfigure parameters in Settings then click ▶ Submit."},

    # ── 结构编辑功能扩展(新增按钮 / 对话框 / 状态栏)─────────────────
    "btn_preset":          {"zh": "📥 载入预设",      "en": "📥 Load Preset"},
    "btn_constraint":      {"zh": "📌 几何约束",      "en": "📌 Constraint"},
    "btn_transform":       {"zh": "↻ 平移/旋转",      "en": "↻ Transform"},
    "struct_empty_hint":   {"zh": "(空结构 — 通过「📥 载入预设」、「📂 载入 .xyz」或「+ 原子」按钮添加内容)",
                            "en": "(empty — load via 📥 Preset, 📂 .xyz file, or + Atom)"},

    # ── 项目管理(新建/打开/导出)─────────────────────────────
    "dlg_new_title":       {"zh": "新建项目",          "en": "New Project"},
    "dlg_new_confirm":     {"zh": "新建项目将丢弃当前所有原子和结果,确定继续吗?",
                            "en": "Creating a new project will discard current atoms and results. Continue?"},
    "dlg_open_proj":       {"zh": "打开项目",          "en": "Open Project"},
    "dlg_export":          {"zh": "导出结果",          "en": "Export Results"},
    "warn_busy":           {"zh": "计算正在运行中,请先中止后再操作。",
                            "en": "A calculation is running. Please abort it first."},
    "warn_no_results":     {"zh": "没有可导出的结果。",  "en": "No results to export."},
    "err_proj_load":       {"zh": "项目加载失败:",      "en": "Failed to load project:"},
    "err_proj_save":       {"zh": "项目保存失败:",      "en": "Failed to save project:"},
    "err_export":          {"zh": "结果导出失败:",      "en": "Failed to export results:"},
    "err_xyz_external":    {"zh": "外部 .xyz 读取失败:", "en": "External .xyz load failed:"},
    "proj_new_done":       {"zh": "已新建空项目。",     "en": "New project created."},
    "proj_loaded":         {"zh": "已加载项目:",        "en": "Loaded project:"},
    "proj_loaded_xyz":     {"zh": "已从 .xyz 加载结构:",
                            "en": "Loaded structure from .xyz:"},
    "warn_xyz_missing":    {"zh": "项目引用的 .xyz 文件不存在:",
                            "en": "The .xyz file referenced by this project is missing:"},
    "warn_xyz_load_empty": {"zh": "已加载空结构,请将 .xyz 文件放回原位置后重新打开项目。",
                            "en": "Loaded an empty structure. Restore the .xyz file and reopen the project."},
    "warn_xyz_write_fail": {"zh": "写入 .xyz 文件失败:", "en": "Failed to write .xyz file:"},

    # ── Stage 流水线(stage_pipeline.py 子进程驱动)──
    "grp_stages":          {"zh": "Stage 流水线(子进程模式)",
                            "en": "Stage Pipeline (subprocess mode)"},
    "stages_hint":         {"zh": "勾选后将通过 subprocess 调用 stage_pipeline.py 执行。"
                                  "选中此区会禁用上方的细粒度流水线(避免冲突)。",
                            "en": "When enabled, runs stage_pipeline.py in a subprocess. "
                                  "This disables the fine-grained pipeline above to avoid conflicts."},
    "chk_stage0":          {"zh": "Stage 0 — 气相参考 / 裸团簇 / 中间体优化与振动",
                            "en": "Stage 0 — gas refs / bare clusters / intermediates"},
    "chk_stage1":          {"zh": "Stage 1 — λ=0 参考 + λ 扫描(腔内优化与振动)",
                            "en": "Stage 1 — λ=0 ref + λ scan (cavity opt + vibration)"},
    "chk_stage2":          {"zh": "Stage 2 — Cu4+CO2* 共振 vs 失谐 ω 扫描",
                            "en": "Stage 2 — Cu4+CO2* on-/off-resonance ω scan"},
    "lbl_stage_workdir":   {"zh": "工作目录:",       "en": "Workdir:"},
    "lbl_stage_script":    {"zh": "脚本路径:",       "en": "Script path:"},
    "warn_no_stages":      {"zh": "请至少勾选一个 Stage。",
                            "en": "Please tick at least one Stage."},
    "worker_starting":     {"zh": "启动子进程...",   "en": "Starting subprocess..."},
    "worker_aborted":      {"zh": "已中止",          "en": "Aborted"},
    "export_done":         {"zh": "已导出结果到:",      "en": "Results exported to:"},
    "lbl_struct_hint2":    {
        "zh": "在表格中编辑坐标 / 切换「XYZ 文本」编写;3D 视图中左键选原子(Shift 多选)、右键拖动单个原子;选 2/3/4 个原子后点「📌 几何约束」编辑键长/键角/二面角。",
        "en": "Edit in table / use the XYZ text tab; left-click to select an atom in 3D (Shift to multi-select), right-drag to move one atom; select 2/3/4 atoms then click 📌 to edit bond / angle / dihedral.",
    },
    "status_pick":         {"zh": "已选中 #{i} {sym}: ({x:+.4f}, {y:+.4f}, {z:+.4f}) Å",
                            "en": "Picked #{i} {sym}: ({x:+.4f}, {y:+.4f}, {z:+.4f}) Å"},
    "status_dist":         {"zh": "距离 #{i}-{j} = {d:.4f} Å",
                            "en": "Distance #{i}-{j} = {d:.4f} Å"},
    "status_angle":        {"zh": "键角 #{i}-{j}-{k} = {a:.3f}°",
                            "en": "Angle #{i}-{j}-{k} = {a:.3f}°"},
    "status_dihedral":     {"zh": "二面角 #{i}-{j}-{k}-{l} = {a:.3f}°",
                            "en": "Dihedral #{i}-{j}-{k}-{l} = {a:.3f}°"},
    "dlg_preset_title":    {"zh": "从 config 载入预设结构", "en": "Load Preset from config"},
    "dlg_preset_intro":    {"zh": "选择 Cu_n 团簇 + 吸附物作为初始构型,载入后可继续编辑。",
                            "en": "Pick Cu_n cluster + adsorbate as initial geometry; you can edit further after loading."},
    "dlg_preset_cluster":  {"zh": "团簇:",                 "en": "Cluster:"},
    "dlg_preset_ads":      {"zh": "吸附物:",               "en": "Adsorbate:"},
    "dlg_preset_bare":     {"zh": "(裸团簇)",             "en": "(bare cluster)"},
    "dlg_ads_title":       {"zh": "放置吸附物",            "en": "Place Adsorbate"},
    "dlg_ads_intro":       {"zh": "以选中原子为吸附位点;未选中时自动使用最高的 Cu 原子。",
                            "en": "Uses the selected atom as the adsorption site; otherwise uses the highest Cu atom."},
    "dlg_ads_species":     {"zh": "吸附物:",               "en": "Adsorbate:"},
    "dlg_ads_builtin":     {"zh": "内置结构",              "en": "Built-in"},
    "dlg_ads_custom":      {"zh": "自定义 XYZ",            "en": "Custom XYZ"},
    "dlg_ads_xyz":         {"zh": "吸附物 XYZ:",           "en": "Adsorbate XYZ:"},
    "dlg_ads_xyz_hint":    {"zh": "输入吸附物自身坐标;放置时会以第一个原子为锚点。",
                            "en": "Enter adsorbate-only coordinates; the first atom is used as the placement anchor."},
    "dlg_ads_height":      {"zh": "高度修正:",             "en": "Height offset:"},
    "dlg_ads_rotate":      {"zh": "绕 z 旋转:",            "en": "Rotate around z:"},
    "dlg_ads_replace":     {"zh": "替换已有非 Cu 吸附物",  "en": "Replace existing non-Cu adsorbate"},
    "dlg_ads_anchor":      {"zh": "吸附位点:",             "en": "Anchor:"},
    "err_no_ads_anchor":   {"zh": "请先载入含 Cu 原子的结构,或选中一个吸附位点原子。",
                            "en": "Load a structure containing Cu, or select an anchor atom first."},
    "dlg_geom_title":      {"zh": "几何约束编辑",          "en": "Geometry Constraint"},
    "dlg_geom_need_select": {"zh": "请先选中 2(键长)/3(键角)/4(二面角)个原子。",
                            "en": "Select 2 (bond) / 3 (angle) / 4 (dihedral) atoms first."},
    "dlg_geom_current":    {"zh": "当前值",                "en": "Current"},
    "dlg_geom_target":     {"zh": "目标值",                "en": "Target"},
    "dlg_geom_move":       {"zh": "移动端点",              "en": "Move endpoint"},
    "dlg_geom_move_j":     {"zh": "移动 j(右端)",         "en": "Move j (second)"},
    "dlg_geom_move_i":     {"zh": "移动 i(左端)",         "en": "Move i (first)"},
    "dlg_geom_move_both":  {"zh": "双向各半",              "en": "Both, half each"},
    "dlg_geom_angle_hint": {"zh": "顶点 = 第 2 个选中原子;旋转第 3 个原子达到目标角。",
                            "en": "Vertex = 2nd selected atom; the 3rd atom is rotated."},
    "dlg_geom_dihedral_hint": {"zh": "绕第 2-3 原子连线旋转第 4 个原子。",
                            "en": "Rotates the 4th atom around the 2nd-3rd bond."},
    "dlg_xform_title":     {"zh": "平移 / 旋转",           "en": "Translate / Rotate"},
    "dlg_xform_translate": {"zh": "平移",                  "en": "Translate"},
    "dlg_xform_rotate":    {"zh": "旋转(绕选区质心)",      "en": "Rotate (around centroid)"},
    "dlg_xform_dxdydz":    {"zh": "ΔX, ΔY, ΔZ:",          "en": "ΔX, ΔY, ΔZ:"},
    "dlg_xform_axis":      {"zh": "轴",                    "en": "Axis"},
    "dlg_xform_target_sel":{"zh": "作用于当前选区({n} 个原子)。",
                            "en": "Applies to current selection ({n} atoms)."},
    "dlg_xform_target_all":{"zh": "选区为空 → 作用于全部 {n} 个原子。",
                            "en": "No selection → applies to all {n} atoms."},
}

def t(key: str) -> str:
    """返回当前语言的翻译字符串。"""
    entry = TR.get(key)
    if entry is None: return key
    return entry.get(LANG, entry.get("zh", key))



def build_geom(cluster: str, ads: str | None) -> str:
    cl = CLUSTERS[cluster]
    tx, ty, tz = cl["top_xyz"]
    geom = cl["geom"]
    if ads:
        key  = "geom_offset_Cu1" if cluster == "Cu1" else "geom_offset"
        rows = []
        for line in ADS_OFFSETS[ads][key].strip().split("\n"):
            p = line.split()
            rows.append(f"{p[0]}  {float(p[1])+tx:.6f}  {float(p[2])+ty:.6f}  {float(p[3])+tz:.6f}")
        geom = geom + "\n" + "\n".join(rows)
    return geom

def calc_spin(cluster: str, ads: str | None) -> int:
    n_cu = len([l for l in CLUSTERS[cluster]["geom"].strip().split("\n")])
    total = n_cu * 11 + (ADS_OFFSETS[ads]["n_electrons"] if ads else 0)
    return total % 2

def parse_geom(geom: str) -> list[tuple]:
    result = []
    for line in geom.strip().split("\n"):
        p = line.split()
        if len(p) >= 4:
            result.append((p[0], float(p[1]), float(p[2]), float(p[3])))
    return result


# ═══════════════════════════════════════════════════════════
#  可视化 Widgets
# ═══════════════════════════════════════════════════════════

class _MplWidget(QWidget):
    """带 matplotlib canvas 的基类。"""
    def __init__(self, figsize=(5,3), parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(0,0,0,0)
        if HAS_MPL:
            self.fig = Figure(figsize=figsize, dpi=96, facecolor="none")
            self.canvas = FigureCanvasQTAgg(self.fig)
            lay.addWidget(self.canvas)
        else:
            lay.addWidget(QLabel("pip install matplotlib"))


class SCFPlotWidget(_MplWidget):
    def __init__(self, parent=None):
        super().__init__((5,3), parent)
        if not HAS_MPL: return
        self.ax_e = self.fig.add_subplot(211)
        self.ax_d = self.fig.add_subplot(212)
        self.fig.subplots_adjust(hspace=0.45,left=0.18,right=0.97,top=0.92,bottom=0.15)
        self._xs, self._ys, self._ds = [], [], []
        self._init_axes()

    def _init_axes(self):
        for ax in (self.ax_e, self.ax_d):
            ax.set_facecolor("#f8f9fa"); ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.4, lw=0.5)
        self.ax_e.set_ylabel("E (Ha)", fontsize=8)
        self.ax_e.set_title("SCF Convergence", fontsize=9, pad=4)
        self.ax_d.set_xlabel("Iter", fontsize=8)
        self.ax_d.set_ylabel("|ΔE|", fontsize=8, color="#e74c3c")
        self.ax_d.set_yscale("log")

    @Slot(int, float, float)
    def add_point(self, i, e, de):
        if not HAS_MPL: return
        self._xs.append(i); self._ys.append(e); self._ds.append(max(de,1e-15))
        for ax in (self.ax_e, self.ax_d): ax.clear()
        self._init_axes()
        self.ax_e.plot(self._xs, self._ys, "o-", c="#2563eb", ms=3, lw=1.5)
        self.ax_d.semilogy(self._xs, self._ds, "s-", c="#e74c3c", ms=3, lw=1.5)
        if self._ys: self.ax_e.set_title(f"SCF  E={self._ys[-1]:.8f} Ha", fontsize=8, pad=4)
        self.canvas.draw_idle()

    def reset(self):
        self._xs.clear(); self._ys.clear(); self._ds.clear()
        if HAS_MPL:
            for ax in (self.ax_e, self.ax_d): ax.clear()
            self._init_axes(); self.canvas.draw_idle()

    def set_label(self, label: str):
        if HAS_MPL:
            self.ax_e.set_title(f"SCF — {label}", fontsize=9, pad=4)
            self.canvas.draw_idle()


class LambdaPlot(_MplWidget):
    """λ 扫描双子图，实时追加。"""
    def __init__(self, parent=None):
        super().__init__((5,3.5), parent)
        if not HAS_MPL: return
        self.ax_e = self.fig.add_subplot(211)
        self.ax_f = self.fig.add_subplot(212)
        self.fig.subplots_adjust(hspace=0.50,left=0.18,right=0.97,top=0.92,bottom=0.12)
        self._data: list[dict] = []
        self._init_axes()

    def _init_axes(self):
        if not HAS_MPL: return
        for ax in (self.ax_e, self.ax_f):
            ax.set_facecolor("#f8f9fa"); ax.tick_params(labelsize=8)
            ax.grid(True, alpha=0.4, lw=0.5)
        self.ax_e.set_xlabel("λ (a.u.)", fontsize=8)
        self.ax_e.set_ylabel("ΔE_ads (eV)", fontsize=8, color="#2563eb")
        self.ax_e.set_title(t("plt_ads_vs_lam"), fontsize=9, pad=4)
        self.ax_f.set_xlabel("λ (a.u.)", fontsize=8)
        self.ax_f.set_ylabel("ν C-O (cm⁻¹)", fontsize=8, color="#8b5cf6")
        self.ax_f.set_title(t("plt_freq_vs_lam"), fontsize=9, pad=4)

    def append_point(self, lam, ads_ev, freq):
        if not HAS_MPL: return
        self._data.append({"l": lam, "e": ads_ev, "f": freq})
        for ax in (self.ax_e, self.ax_f): ax.clear()
        self._init_axes()
        le = [d["l"] for d in self._data if d["e"] is not None]
        ee = [d["e"] for d in self._data if d["e"] is not None]
        lf = [d["l"] for d in self._data if d["f"] is not None]
        ff = [d["f"] for d in self._data if d["f"] is not None]
        if le: self.ax_e.plot(le, ee, "o-", c="#2563eb", ms=4, lw=1.5)
        if lf: self.ax_f.plot(lf, ff, "s-", c="#8b5cf6", ms=4, lw=1.5)
        self.canvas.draw_idle()

    def reset(self):
        self._data.clear()
        if HAS_MPL:
            for ax in (self.ax_e, self.ax_f): ax.clear()
            self._init_axes(); self.canvas.draw_idle()


class FreqBarPlot(_MplWidget):
    """腔诱导频移柱状图。"""
    def __init__(self, parent=None):
        super().__init__((5,2.5), parent)
        if not HAS_MPL: return
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.14,right=0.97,top=0.88,bottom=0.25)
        self._init_axes()

    def _init_axes(self):
        if not HAS_MPL: return
        self.ax.set_facecolor("#f8f9fa"); self.ax.tick_params(labelsize=8)
        self.ax.grid(True, axis="y", alpha=0.4)
        self.ax.set_ylabel("Δω (cm⁻¹)", fontsize=8)
        self.ax.set_title(t("plt_shift_title"), fontsize=9, pad=4)

    def update_data(self, labels, shifts):
        if not HAS_MPL: return
        self.ax.clear(); self._init_axes()
        colors = ["#50fa7b" if s >= 0 else "#ff5555" for s in shifts]
        self.ax.bar(labels, shifts, color=colors, edgecolor="#44475a", lw=0.5)
        self.ax.axhline(0, color="#555555", lw=0.8)
        self.ax.tick_params(axis="x", rotation=20, labelsize=7)
        self.canvas.draw_idle()


# ═══════════════════════════════════════════════════════════
#  3D 结构预览
# ═══════════════════════════════════════════════════════════

ACOLORS = {"H":"#fff","C":"#404040","N":"#3050f8","O":"#ff0d0d","F":"#90e050",
           "P":"#ff8000","S":"#ffff30","Cl":"#1ff01f","Br":"#a62929","I":"#940094",
           "Si":"#f0c8a0","Fe":"#e06633","Ca":"#3dff00","Mg":"#8aff00",
           "Na":"#ab5cf2","K":"#8f40d4","Zn":"#7d80b0","Cu":"#c88033",
           "Al":"#bfa6a6","Ti":"#bfc2c7"}
ARADII  = {"H":0.31,"C":0.77,"N":0.75,"O":0.73,"F":0.71,"P":1.06,"S":1.02,
           "Cl":0.99,"Br":1.14,"I":1.33,"Si":1.11,"Fe":1.32,"Ca":1.74,
           "Mg":1.30,"Na":1.54,"K":1.96,"Cu":1.28}

def _fit_3d_axes(ax, atoms, pad=0.8):
    import numpy as np
    if not atoms:
        return
    coords = np.array([[a[1], a[2], a[3]] for a in atoms], dtype=float)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    center = (lo + hi) / 2.0
    span = float(max(np.max(hi - lo), 1.0) + 2 * pad)
    half = span / 2.0
    ax.set_xlim3d(center[0] - half, center[0] + half)
    ax.set_ylim3d(center[1] - half, center[1] + half)
    ax.set_zlim3d(center[2] - half, center[2] + half)

def _zoom_3d_axes(ax, factor):
    for getter, setter in [
        (ax.get_xlim3d, ax.set_xlim3d),
        (ax.get_ylim3d, ax.set_ylim3d),
        (ax.get_zlim3d, ax.set_zlim3d),
    ]:
        lo, hi = getter()
        c = (lo + hi) / 2.0
        half = (hi - lo) * factor / 2.0
        setter(c - half, c + half)

class Structure3DWidget(_MplWidget):
    def __init__(self, parent=None):
        super().__init__((5,4), parent)
        if not HAS_MPL: return
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
        self._atoms: list[tuple] = []
        self._had_structure = False
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self._placeholder()

    def _placeholder(self):
        self.ax.clear(); self.ax.set_facecolor("#1a1a2e"); self.fig.set_facecolor("#1a1a2e")
        self.ax.text(0.5,0.5,0.5,t("plt_3d_placeholder"),
                     ha="center",va="center",color="#555555",fontsize=11,transform=self.ax.transAxes)
        self.ax.set_axis_off(); self.canvas.draw_idle()

    def update_structure(self, atoms, reset_view=False):
        if not HAS_MPL:
            return
        had_structure = self._had_structure
        self._atoms = list(atoms) if atoms else []
        if self._atoms:
            self._redraw(reset_view=reset_view or not had_structure)
            self._had_structure = True
        else:
            self._had_structure = False
            self._placeholder()

    def _redraw(self, reset_view=False):
        import numpy as np
        self.ax.clear(); self.ax.set_facecolor("#1a1a2e"); self.fig.set_facecolor("#1a1a2e")
        self.ax.set_axis_on()
        a = self._atoms
        for i in range(len(a)):
            for j in range(i+1,len(a)):
                s1,x1,y1,z1=a[i]; s2,x2,y2,z2=a[j]
                d = np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
                if d < (ARADII.get(s1.capitalize(),0.8)+ARADII.get(s2.capitalize(),0.8))*1.3:
                    self.ax.plot([x1,x2],[y1,y2],[z1,z2],color="#888",lw=2,alpha=0.7,zorder=1)
        for sym,x,y,z in a:
            k=sym.capitalize(); c=ACOLORS.get(k,"#aaa"); r=ARADII.get(k,0.8)
            self.ax.scatter([x],[y],[z],s=max(40,min(300,r*200)),c=c,
                            edgecolors="#fff",linewidths=0.5,depthshade=True,zorder=2)
            self.ax.text(x,y,z+r*0.5+0.1,k,fontsize=9,ha="center",va="bottom",color="#f8f8f2",zorder=3)
        for attr,lbl in zip(["x","y","z"],["X (Å)","Y (Å)","Z (Å)"]):
            getattr(self.ax,f"set_{attr}label")(lbl,fontsize=10,color="#9ca3af",labelpad=8)
        self.ax.tick_params(colors="#9ca3af",labelsize=9)
        for pane in [self.ax.xaxis.pane,self.ax.yaxis.pane,self.ax.zaxis.pane]:
            pane.fill=False; pane.set_edgecolor("#44475a")
        self.ax.grid(True,alpha=0.2,color="#44475a")
        self.ax.set_title(t("plt_n_atoms").format(n=len(a)),fontsize=10,color="#d1d5db",pad=6)
        if reset_view:
            _fit_3d_axes(self.ax, a)
        self.fig.canvas.draw()
        self.canvas.draw_idle()

    def _on_scroll(self, event):
        if not self._atoms or event.inaxes is not self.ax:
            return
        _zoom_3d_axes(self.ax, 0.85 if event.button == "up" else 1.15)
        self.canvas.draw_idle()

    def reset(self):
        self._atoms=[]
        self._had_structure = False
        if HAS_MPL: self._placeholder()


# ═══════════════════════════════════════════════════════════
#  结构编辑功能 (拾取/拖拽 + 几何约束 + 平移旋转 + 预设载入)
# ═══════════════════════════════════════════════════════════

# ── 几何工具(纯函数,不依赖外部模块)─────────────────────

def _bond_length(p1, p2) -> float:
    import numpy as np
    return float(np.linalg.norm(np.asarray(p2) - np.asarray(p1)))

def _bond_angle(p1, p2, p3) -> float:
    """∠p1-p2-p3, 顶点 p2,返回度。"""
    import numpy as np
    v1 = np.asarray(p1) - np.asarray(p2)
    v2 = np.asarray(p3) - np.asarray(p2)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    c = np.clip(v1 @ v2 / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))

def _dihedral(p1, p2, p3, p4) -> float:
    """二面角 IUPAC 右手定则,范围 [-180, 180]。"""
    import numpy as np
    b1 = np.asarray(p2) - np.asarray(p1)
    b2 = np.asarray(p3) - np.asarray(p2)
    b3 = np.asarray(p4) - np.asarray(p3)
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    b2n = b2 / max(np.linalg.norm(b2), 1e-12)
    m = np.cross(n1, b2n)
    return float(np.degrees(np.arctan2(m @ n2, n1 @ n2)))

def _rot_matrix(axis, theta_rad: float):
    """Rodrigues 旋转矩阵(右手系正旋)。"""
    import numpy as np, math
    axis = np.asarray(axis, dtype=float)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    a = math.cos(theta_rad / 2.0)
    b, c, d = -axis * math.sin(theta_rad / 2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([
        [aa+bb-cc-dd, 2*(bc+ad),   2*(bd-ac)  ],
        [2*(bc-ad),   aa+cc-bb-dd, 2*(cd+ab)  ],
        [2*(bd+ac),   2*(cd-ab),   aa+dd-bb-cc],
    ])

def _atoms_to_array(atoms):
    import numpy as np
    syms   = [a[0] for a in atoms]
    coords = np.array([[a[1], a[2], a[3]] for a in atoms], dtype=float)
    return syms, coords

def _array_to_atoms(syms, coords):
    return [(syms[i], float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2]))
            for i in range(len(syms))]

def _set_bond_length(coords, i, j, target, move="j"):
    import numpy as np
    coords = coords.copy()
    v = coords[j] - coords[i]
    d = np.linalg.norm(v)
    if d < 1e-9:
        return coords
    u = v / d
    delta = target - d
    if move == "j":
        coords[j] = coords[j] + u * delta
    elif move == "i":
        coords[i] = coords[i] - u * delta
    else:  # both
        coords[j] = coords[j] + u * (delta / 2)
        coords[i] = coords[i] - u * (delta / 2)
    return coords

def _set_bond_angle(coords, i, j, k, target_deg):
    import numpy as np, math
    coords = coords.copy()
    cur = _bond_angle(coords[i], coords[j], coords[k])
    if math.isnan(cur):
        return coords
    delta = math.radians(target_deg - cur)
    v_ji = coords[i] - coords[j]
    v_jk = coords[k] - coords[j]
    axis = np.cross(v_ji, v_jk)
    if np.linalg.norm(axis) < 1e-9:
        # 三点共线 fallback
        axis = np.cross(v_ji, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-9:
            axis = np.cross(v_ji, np.array([0.0, 1.0, 0.0]))
    R = _rot_matrix(axis, delta)
    coords[k] = R @ (coords[k] - coords[j]) + coords[j]
    return coords

def _set_dihedral(coords, i, j, k, l, target_deg):
    """注意:_rot_matrix 是右手 Rodrigues, 与 IUPAC 二面角定义差一个负号。"""
    import numpy as np, math
    coords = coords.copy()
    cur = _dihedral(coords[i], coords[j], coords[k], coords[l])
    if math.isnan(cur):
        return coords
    delta = math.radians(target_deg - cur)
    axis = coords[k] - coords[j]
    R = _rot_matrix(axis, -delta)
    coords[l] = R @ (coords[l] - coords[k]) + coords[k]
    return coords

def _translate_indices(coords, idxs, vec):
    import numpy as np
    coords = coords.copy()
    for i in idxs:
        coords[i] = coords[i] + np.asarray(vec)
    return coords

def _rotate_indices(coords, idxs, axis, theta_deg, center=None):
    import numpy as np, math
    coords = coords.copy()
    if not idxs:
        return coords
    if center is None:
        center = np.mean([coords[i] for i in idxs], axis=0)
    R = _rot_matrix(axis, math.radians(theta_deg))
    for i in idxs:
        coords[i] = R @ (coords[i] - center) + center
    return coords


# ── 可拾取的 3D 结构视图(Structure3DWidget 的增强版)──────

class PickableStructure3DWidget(_MplWidget):
    """
    可拾取/可拖拽的 3D 结构视图,接口与 Structure3DWidget 兼容。

    交互:
      左键单击原子 → 选中(Shift/Ctrl 多选)
      左键单击空白 → 清选区
      左键拖动空白 → mpl 默认旋转视角
      右键拖动原子 → 平移单个原子
      滚轮         → mpl 默认缩放
    """

    selection_changed = Signal(set)
    atoms_dragged     = Signal(list)

    def __init__(self, parent=None):
        super().__init__((5, 4), parent)
        if not HAS_MPL:
            return
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self._atoms: list[tuple] = []
        self._selected: set[int] = set()

        # 拖拽状态
        self._drag_idx = None
        self._drag_z_screen = 0.0
        self._press_xy = None
        self._press_idx = None
        self._drag_active = False
        self._had_structure = False

        self.canvas.mpl_connect("button_press_event",   self._on_press)
        self.canvas.mpl_connect("motion_notify_event",  self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("scroll_event",         self._on_scroll)

        self._placeholder()

    # ── 公共 API(兼容 Structure3DWidget)─────────────────────
    def update_structure(self, atoms, reset_view=False):
        if not HAS_MPL: return
        had_structure = self._had_structure
        self._atoms = list(atoms) if atoms else []
        # 选区索引可能越界
        self._selected = {i for i in self._selected if i < len(self._atoms)}
        if self._atoms:
            self._redraw(reset_view=reset_view or not had_structure)
            self._had_structure = True
        else:
            self._had_structure = False
            self._placeholder()

    def reset(self):
        self._atoms = []
        self._selected.clear()
        self._had_structure = False
        if HAS_MPL: self._placeholder()

    # ── 选区 ────────────────────────────────────────────────
    def get_selection(self) -> set:
        return set(self._selected)

    def set_selection(self, indices: set):
        new = {i for i in indices if 0 <= i < len(self._atoms)}
        if new != self._selected:
            self._selected = new
            if self._atoms: self._redraw()
            self.selection_changed.emit(set(new))

    def clear_selection(self):
        if self._selected:
            self._selected.clear()
            if self._atoms: self._redraw()
            self.selection_changed.emit(set())

    # ── 屏幕 ↔ 世界坐标 ─────────────────────────────────────
    def _world_to_screen(self, xyz):
        from mpl_toolkits.mplot3d import proj3d
        x2, y2, z2 = proj3d.proj_transform(xyz[0], xyz[1], xyz[2],
                                           self.ax.get_proj())
        sx, sy = self.ax.transData.transform((x2, y2))
        return sx, sy, z2

    def _screen_to_world(self, sx, sy, z2):
        import numpy as np
        x2, y2 = self.ax.transData.inverted().transform((sx, sy))
        M = self.ax.get_proj()
        try:
            from mpl_toolkits.mplot3d.proj3d import inv_transform
            xw, yw, zw = inv_transform(x2, y2, z2, M)
        except ImportError:
            inv = np.linalg.inv(M)
            v = inv @ np.array([x2, y2, z2, 1.0])
            xw, yw, zw = v[0]/v[3], v[1]/v[3], v[2]/v[3]
        return np.array([xw, yw, zw])

    def _pick_atom(self, sx, sy, radius_px=14):
        import math
        if not self._atoms:
            return None
        best, best_d = None, radius_px
        for i, (sym, x, y, z) in enumerate(self._atoms):
            try:
                px, py, _ = self._world_to_screen((x, y, z))
            except Exception:
                continue
            d = math.hypot(px - sx, py - sy)
            if d < best_d:
                best, best_d = i, d
        return best

    # ── 鼠标事件 ────────────────────────────────────────────
    def _on_press(self, event):
        if event.inaxes is not self.ax or event.x is None:
            return
        # 右键 = 拖原子
        if event.button == 3:
            idx = self._pick_atom(event.x, event.y)
            if idx is not None:
                self._drag_idx = idx
                _, _, z_screen = self._world_to_screen(self._atoms[idx][1:4])
                self._drag_z_screen = z_screen
                self._drag_active = False
                self._press_xy = (event.x, event.y)
                try:
                    self.ax.disable_mouse_rotation()
                except Exception:
                    pass
            return
        # 左键: 留给 release 判定 click vs 旋转
        if event.button == 1:
            self._press_xy = (event.x, event.y)
            self._press_idx = self._pick_atom(event.x, event.y)

    def _on_motion(self, event):
        import math
        if self._drag_idx is None or event.inaxes is not self.ax:
            return
        if event.x is None or event.y is None:
            return
        if not self._drag_active:
            sx0, sy0 = self._press_xy
            if math.hypot(event.x - sx0, event.y - sy0) < 3:
                return
            self._drag_active = True
        new_xyz = self._screen_to_world(event.x, event.y, self._drag_z_screen)
        sym, _, _, _ = self._atoms[self._drag_idx]
        self._atoms[self._drag_idx] = (sym, float(new_xyz[0]),
                                        float(new_xyz[1]), float(new_xyz[2]))
        self._redraw()

    def _on_release(self, event):
        import math
        # 右键释放 → 提交拖拽结果
        if event.button == 3 and self._drag_idx is not None:
            try:
                self.ax.mouse_init()
            except Exception:
                pass
            if self._drag_active:
                self.atoms_dragged.emit(list(self._atoms))
            self._drag_idx = None
            self._drag_active = False
            self._press_xy = None
            return
        # 左键释放 → click 还是旋转
        if event.button == 1 and self._press_xy is not None:
            sx0, sy0 = self._press_xy
            if event.x is None:
                self._press_xy = None
                self._press_idx = None
                return
            moved = math.hypot(event.x - sx0, event.y - sy0)
            if moved < 3:
                if self._press_idx is not None:
                    mods = QApplication.keyboardModifiers()
                    if mods & (Qt.ShiftModifier | Qt.ControlModifier):
                        new_sel = set(self._selected)
                        if self._press_idx in new_sel:
                            new_sel.discard(self._press_idx)
                        else:
                            new_sel.add(self._press_idx)
                        self.set_selection(new_sel)
                    else:
                        self.set_selection({self._press_idx})
                else:
                    self.clear_selection()
            self._press_xy = None
            self._press_idx = None

    def _on_scroll(self, event):
        if not self._atoms or event.inaxes is not self.ax:
            return
        _zoom_3d_axes(self.ax, 0.85 if event.button == "up" else 1.15)
        self.canvas.draw_idle()

    # ── 绘制(沿用 Structure3DWidget 风格,加选区高亮)─────
    def _placeholder(self):
        self.ax.clear()
        self.ax.set_facecolor("#1a1a2e")
        self.fig.set_facecolor("#1a1a2e")
        self.ax.text(0.5, 0.5, 0.5, t("plt_3d_placeholder"),
                     ha="center", va="center", color="#555555",
                     fontsize=11, transform=self.ax.transAxes)
        self.ax.set_axis_off()
        self.canvas.draw_idle()

    def _redraw(self, reset_view=False):
        import numpy as np, math
        # 保留视角(避免每次重绘视角跳)
        elev, azim = self.ax.elev, self.ax.azim
        had_data = self._had_structure and not reset_view
        prev_xlim = self.ax.get_xlim3d() if had_data else None
        prev_ylim = self.ax.get_ylim3d() if had_data else None
        prev_zlim = self.ax.get_zlim3d() if had_data else None

        self.ax.clear()
        self.ax.set_facecolor("#1a1a2e")
        self.fig.set_facecolor("#1a1a2e")
        self.ax.set_axis_on()

        a = self._atoms
        # 键
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                s1, x1, y1, z1 = a[i]
                s2, x2, y2, z2 = a[j]
                d = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
                if d < (ARADII.get(s1.capitalize(), 0.8)
                        + ARADII.get(s2.capitalize(), 0.8)) * 1.3:
                    self.ax.plot([x1, x2], [y1, y2], [z1, z2],
                                 color="#888", lw=2, alpha=0.7, zorder=1)
        # 原子
        for idx, (sym, x, y, z) in enumerate(a):
            k = sym.capitalize()
            c = ACOLORS.get(k, "#aaa")
            r = ARADII.get(k, 0.8)
            sel = idx in self._selected
            edge   = "#00ffff" if sel else "#fff"
            edge_w = 2.5       if sel else 0.5
            self.ax.scatter([x], [y], [z],
                            s=max(40, min(300, r * 200)),
                            c=c, edgecolors=edge, linewidths=edge_w,
                            depthshade=True, zorder=2)
            label = f"{k}{idx}" if sel else k
            self.ax.text(x, y, z + r*0.5 + 0.1, label,
                         fontsize=9, ha="center", va="bottom",
                         color="#ffd700" if sel else "#f8f8f2",
                         zorder=3)
        for attr, lbl in zip(["x", "y", "z"], ["X (Å)", "Y (Å)", "Z (Å)"]):
            getattr(self.ax, f"set_{attr}label")(lbl, fontsize=10, color="#9ca3af", labelpad=8)
        self.ax.tick_params(colors="#9ca3af", labelsize=9)
        for pane in [self.ax.xaxis.pane, self.ax.yaxis.pane, self.ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor("#44475a")
        self.ax.grid(True, alpha=0.2, color="#44475a")
        self.ax.set_title(t("plt_n_atoms").format(n=len(a)),
                          fontsize=10, color="#d1d5db", pad=6)

        if prev_xlim and len(a) > 0:
            self.ax.set_xlim3d(prev_xlim)
            self.ax.set_ylim3d(prev_ylim)
            self.ax.set_zlim3d(prev_zlim)
        else:
            _fit_3d_axes(self.ax, a)
        self.ax.view_init(elev=elev, azim=azim)
        self.fig.canvas.draw()
        self.canvas.draw_idle()


# ── 对话框 1: 从 CLUSTERS / ADS_OFFSETS 载入预设 ──────────

class PresetLoaderDialog(QDialog):
    """选 (cluster, adsorbate),通过 build_geom 生成 atoms。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("dlg_preset_title"))
        self.resize(420, 220)

        v = QVBoxLayout(self)
        v.addWidget(QLabel(t("dlg_preset_intro")))

        form = QFormLayout()
        self.cb_cluster = QComboBox()
        self.cb_ads     = QComboBox()
        self.cb_cluster.addItems(list(CLUSTERS.keys()))
        self.cb_ads.addItem(t("dlg_preset_bare"), userData=None)
        for name in ADS_OFFSETS.keys():
            self.cb_ads.addItem(name, userData=name)
        form.addRow(t("dlg_preset_cluster"), self.cb_cluster)
        form.addRow(t("dlg_preset_ads"),     self.cb_ads)
        v.addLayout(form)

        self.lbl_info = QLabel("")
        self.lbl_info.setStyleSheet("color:#88bbdd;font-size:9pt;")
        self.lbl_info.setWordWrap(True)
        v.addWidget(self.lbl_info)

        self.cb_cluster.currentIndexChanged.connect(self._update_info)
        self.cb_ads.currentIndexChanged.connect(self._update_info)
        self._update_info()

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def _update_info(self):
        cl  = self.cb_cluster.currentText()
        ads = self.cb_ads.currentData()
        info_cl = CLUSTERS.get(cl, {})
        n_cu = len([l for l in info_cl.get("geom", "").strip().splitlines()])
        msg = f"{cl}: {n_cu} Cu, {info_cl.get('method', '?')}"
        if ads:
            ai = ADS_OFFSETS.get(ads, {})
            msg += (f"  +  {ads}  (n_e={ai.get('n_electrons','?')}, "
                    f"ν={ai.get('nu_target_cm','?')} cm⁻¹)")
        self.lbl_info.setText(msg)

    def selection(self):
        return self.cb_cluster.currentText(), self.cb_ads.currentData()

    def atoms(self):
        cl, ads = self.selection()
        return parse_geom(build_geom(cl, ads))


class ElementDelegate(QStyledItemDelegate):
    """Dropdown editor for the element column in the coordinate table."""

    def createEditor(self, parent, option, index):
        if index.column() == 0:
            cb = QComboBox(parent)
            cb.addItems(COMMON_ELEMENTS)
            cb.setEditable(True)
            return cb
        return super().createEditor(parent, option, index)

    def setEditorData(self, editor, index):
        if isinstance(editor, QComboBox):
            text = index.data() or ""
            i = editor.findText(text)
            if i >= 0:
                editor.setCurrentIndex(i)
            else:
                editor.setEditText(text)
            return
        super().setEditorData(editor, index)

    def setModelData(self, editor, model, index):
        if isinstance(editor, QComboBox):
            model.setData(index, editor.currentText().strip())
            return
        super().setModelData(editor, model, index)


class AdsorbatePlacementDialog(QDialog):
    """Place a built-in adsorbate on the selected/auto-detected surface atom."""

    def __init__(self, atoms, selected, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("dlg_ads_title"))
        self.resize(560, 420)
        self._atoms = list(atoms)
        self._sel = sorted(selected)
        self._result_atoms = list(atoms)
        self._anchor = self._find_anchor()

        v = QVBoxLayout(self)
        intro = QLabel(t("dlg_ads_intro"))
        intro.setWordWrap(True)
        v.addWidget(intro)

        form = QFormLayout()
        self.cb_mode = QComboBox()
        self.cb_mode.addItems([t("dlg_ads_builtin"), t("dlg_ads_custom")])
        form.addRow(t("dlg_ads_species"), self.cb_mode)

        self.cb_ads = QComboBox()
        self.cb_ads.addItems(list(ADS_OFFSETS.keys()))
        form.addRow("", self.cb_ads)

        self.edit_xyz = QPlainTextEdit()
        self.edit_xyz.setFont(QFont("Courier New", 9))
        self.edit_xyz.setMinimumHeight(110)
        self.edit_xyz.setPlainText(self._default_custom_xyz())
        self.edit_xyz.setPlaceholderText(t("dlg_ads_xyz_hint"))
        form.addRow(t("dlg_ads_xyz"), self.edit_xyz)

        self.spin_height = QDoubleSpinBox()
        self.spin_height.setRange(-5.0, 5.0)
        self.spin_height.setDecimals(3)
        self.spin_height.setSingleStep(0.05)
        self.spin_height.setSuffix(" Å")
        form.addRow(t("dlg_ads_height"), self.spin_height)

        self.spin_rot = QDoubleSpinBox()
        self.spin_rot.setRange(-180.0, 180.0)
        self.spin_rot.setDecimals(2)
        self.spin_rot.setSingleStep(5.0)
        self.spin_rot.setSuffix(" °")
        form.addRow(t("dlg_ads_rotate"), self.spin_rot)

        self.chk_replace = QCheckBox(t("dlg_ads_replace"))
        self.chk_replace.setChecked(True)
        form.addRow("", self.chk_replace)
        v.addLayout(form)

        self.lbl_anchor = QLabel()
        self.lbl_anchor.setStyleSheet("color:#88bbdd;font-size:9pt;")
        v.addWidget(self.lbl_anchor)
        self._update_anchor_label()
        self.cb_mode.currentIndexChanged.connect(self._update_mode)
        self.cb_ads.currentIndexChanged.connect(self._update_builtin_xyz)
        self._update_mode()
        v.addStretch(1)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._on_ok)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def _find_anchor(self):
        if self._sel:
            i = self._sel[0]
            if 0 <= i < len(self._atoms):
                return i, self._atoms[i]
        cu = [(i, a) for i, a in enumerate(self._atoms)
              if str(a[0]).capitalize() == "Cu"]
        if cu:
            return max(cu, key=lambda ia: ia[1][3])
        return None

    def _update_anchor_label(self):
        if not self._anchor:
            self.lbl_anchor.setText(t("err_no_ads_anchor"))
            return
        i, (sym, x, y, z) = self._anchor
        self.lbl_anchor.setText(
            f"{t('dlg_ads_anchor')} #{i} {sym} ({x:+.3f}, {y:+.3f}, {z:+.3f}) Å")

    def _default_custom_xyz(self):
        ads = next(iter(ADS_OFFSETS.keys()))
        return self._builtin_xyz(ads)

    def _builtin_xyz(self, ads_name):
        key = "geom_offset"
        rows = ADS_OFFSETS[ads_name][key].strip().splitlines()
        out = [str(len(rows)), ads_name]
        for line in rows:
            sym, x, y, z = line.split()
            out.append(f"{sym:<4s} {float(x):10.5f} {float(y):10.5f} {float(z):10.5f}")
        return "\n".join(out)

    @Slot()
    def _update_mode(self):
        custom = self.cb_mode.currentIndex() == 1
        self.cb_ads.setVisible(not custom)
        self.edit_xyz.setVisible(custom)

    @Slot()
    def _update_builtin_xyz(self):
        if self.cb_mode.currentIndex() == 0:
            return
        self.edit_xyz.setPlainText(self._builtin_xyz(self.cb_ads.currentText()))

    def _ads_atoms(self, anchor):
        import math
        _, (_, ax, ay, az) = anchor
        n_cu = sum(1 for a in self._atoms if str(a[0]).capitalize() == "Cu")
        if self.cb_mode.currentIndex() == 0:
            ads_name = self.cb_ads.currentText()
            key = "geom_offset_Cu1" if n_cu <= 1 and "geom_offset_Cu1" in ADS_OFFSETS[ads_name] else "geom_offset"
            raw_atoms = parse_geom(ADS_OFFSETS[ads_name][key])
        else:
            raw_atoms = StructurePanel._parse_xyz_text(self.edit_xyz.toPlainText())
        if not raw_atoms:
            raise ValueError("No adsorbate atoms")
        ox, oy, oz = raw_atoms[0][1:4]
        rows = []
        theta = math.radians(self.spin_rot.value())
        ct, st = math.cos(theta), math.sin(theta)
        dz = self.spin_height.value()
        for sym, x, y, z in raw_atoms:
            x0, y0, z0 = float(x) - ox, float(y) - oy, float(z) - oz
            xr = x0 * ct - y0 * st
            yr = x0 * st + y0 * ct
            rows.append((sym, ax + xr, ay + yr, az + z0 + dz))
        return rows

    @Slot()
    def _on_ok(self):
        if not self._anchor:
            QMessageBox.warning(self, t("dlg_ads_title"), t("err_no_ads_anchor"))
            return
        base = list(self._atoms)
        has_cu = any(str(a[0]).capitalize() == "Cu" for a in base)
        if self.chk_replace.isChecked() and has_cu:
            base = [a for a in base if str(a[0]).capitalize() == "Cu"]
        try:
            self._result_atoms = base + self._ads_atoms(self._anchor)
        except Exception as ex:
            QMessageBox.warning(self, t("dlg_ads_title"), f"{type(ex).__name__}: {ex}")
            return
        self.accept()

    def result_atoms(self):
        return self._result_atoms


# ── 对话框 2: 几何约束(键长/键角/二面角)─────────────────

class GeomConstraintDialog(QDialog):
    """根据选区原子数自动切换:2→键长, 3→键角, 4→二面角。"""
    def __init__(self, atoms, selected, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("dlg_geom_title"))
        self.resize(420, 260)

        self._atoms = list(atoms)
        self._sel   = sorted(selected)
        self._result_atoms = list(atoms)

        v = QVBoxLayout(self)
        n = len(self._sel)
        if n not in (2, 3, 4):
            v.addWidget(QLabel(t("dlg_geom_need_select")))
            bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
            bb.rejected.connect(self.reject)
            v.addWidget(bb)
            self.spin = None
            return

        # 选中原子描述
        info_lines = []
        for i in self._sel:
            sym, x, y, z = self._atoms[i]
            info_lines.append(f"#{i} {sym}: ({x:+.4f}, {y:+.4f}, {z:+.4f})")
        info = QLabel("\n".join(info_lines))
        info.setStyleSheet("color:#88bbdd;font-family:Courier;font-size:9pt;")
        v.addWidget(info)

        form = QFormLayout()
        syms, coords = _atoms_to_array(self._atoms)
        self.cb_move = None

        if n == 2:
            i, j = self._sel
            cur = _bond_length(coords[i], coords[j])
            form.addRow(t("dlg_geom_current"),
                        QLabel(f"d({i}-{j}) = {cur:.4f} Å"))
            self.spin = QDoubleSpinBox()
            self.spin.setRange(0.05, 50.0); self.spin.setDecimals(4)
            self.spin.setSingleStep(0.05); self.spin.setSuffix(" Å")
            self.spin.setValue(cur)
            form.addRow(t("dlg_geom_target"), self.spin)
            self.cb_move = QComboBox()
            self.cb_move.addItems([t("dlg_geom_move_j"),
                                   t("dlg_geom_move_i"),
                                   t("dlg_geom_move_both")])
            form.addRow(t("dlg_geom_move"), self.cb_move)
        elif n == 3:
            i, j, k = self._sel
            cur = _bond_angle(coords[i], coords[j], coords[k])
            form.addRow(t("dlg_geom_current"),
                        QLabel(f"∠({i}-{j}-{k}) = {cur:.3f}°"))
            self.spin = QDoubleSpinBox()
            self.spin.setRange(0.0, 180.0); self.spin.setDecimals(3)
            self.spin.setSingleStep(1.0); self.spin.setSuffix(" °")
            self.spin.setValue(cur)
            form.addRow(t("dlg_geom_target"), self.spin)
            v.addWidget(QLabel(t("dlg_geom_angle_hint")))
        else:  # n == 4
            i, j, k, l = self._sel
            cur = _dihedral(coords[i], coords[j], coords[k], coords[l])
            form.addRow(t("dlg_geom_current"),
                        QLabel(f"φ({i}-{j}-{k}-{l}) = {cur:.3f}°"))
            self.spin = QDoubleSpinBox()
            self.spin.setRange(-180.0, 180.0); self.spin.setDecimals(3)
            self.spin.setSingleStep(1.0); self.spin.setSuffix(" °")
            self.spin.setValue(cur)
            form.addRow(t("dlg_geom_target"), self.spin)
            v.addWidget(QLabel(t("dlg_geom_dihedral_hint")))

        v.addLayout(form)
        v.addStretch(1)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._on_ok)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    @Slot()
    def _on_ok(self):
        if self.spin is None:
            self.reject(); return
        syms, coords = _atoms_to_array(self._atoms)
        n = len(self._sel)
        try:
            if n == 2:
                i, j = self._sel
                move_map = {0: "j", 1: "i", 2: "both"}
                coords = _set_bond_length(
                    coords, i, j, self.spin.value(),
                    move=move_map[self.cb_move.currentIndex()])
            elif n == 3:
                coords = _set_bond_angle(coords, *self._sel,
                                          target_deg=self.spin.value())
            elif n == 4:
                coords = _set_dihedral(coords, *self._sel,
                                        target_deg=self.spin.value())
        except Exception as e:
            QMessageBox.warning(self, t("dlg_geom_title"),
                                f"{type(e).__name__}: {e}")
            return
        self._result_atoms = _array_to_atoms(syms, coords)
        self.accept()

    def result_atoms(self):
        return self._result_atoms


# ── 对话框 3: 整体平移 / 旋转 ─────────────────────────────

class TransformDialog(QDialog):
    """对选区(空选区时全部)做平移 + 绕轴旋转。"""
    def __init__(self, atoms, selected, parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("dlg_xform_title"))
        self.resize(440, 320)

        self._atoms = list(atoms)
        self._sel   = list(selected) if selected else list(range(len(atoms)))
        self._result_atoms = list(atoms)

        v = QVBoxLayout(self)
        msg = (t("dlg_xform_target_sel").format(n=len(self._sel))
               if selected else
               t("dlg_xform_target_all").format(n=len(atoms)))
        v.addWidget(QLabel(msg))

        # 平移
        gb_t = QGroupBox(t("dlg_xform_translate"))
        ft = QFormLayout(gb_t)
        h_t = QHBoxLayout()
        self.dx = self._make_spin(); self.dy = self._make_spin(); self.dz = self._make_spin()
        for s in (self.dx, self.dy, self.dz):
            h_t.addWidget(s)
        wt = QWidget(); wt.setLayout(h_t)
        ft.addRow(t("dlg_xform_dxdydz"), wt)
        v.addWidget(gb_t)

        # 旋转
        gb_r = QGroupBox(t("dlg_xform_rotate"))
        fr = QFormLayout(gb_r)
        h_r = QHBoxLayout()
        self.cb_axis = QComboBox(); self.cb_axis.addItems(["X", "Y", "Z"])
        self.theta = QDoubleSpinBox()
        self.theta.setRange(-360.0, 360.0); self.theta.setDecimals(3)
        self.theta.setSingleStep(5.0); self.theta.setSuffix(" °")
        h_r.addWidget(QLabel(t("dlg_xform_axis")))
        h_r.addWidget(self.cb_axis)
        h_r.addWidget(QLabel("θ:"))
        h_r.addWidget(self.theta)
        wr = QWidget(); wr.setLayout(h_r)
        fr.addRow(wr)
        v.addWidget(gb_r)

        v.addStretch(1)

        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                              | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._on_ok)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def _make_spin(self):
        s = QDoubleSpinBox()
        s.setRange(-100.0, 100.0); s.setDecimals(4)
        s.setSingleStep(0.1); s.setSuffix(" Å")
        return s

    @Slot()
    def _on_ok(self):
        import numpy as np
        syms, coords = _atoms_to_array(self._atoms)
        vec = np.array([self.dx.value(), self.dy.value(), self.dz.value()])
        if np.linalg.norm(vec) > 1e-12:
            coords = _translate_indices(coords, self._sel, vec)
        if abs(self.theta.value()) > 1e-12:
            ax_map = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]}
            ax = np.array(ax_map[self.cb_axis.currentIndex()], dtype=float)
            coords = _rotate_indices(coords, self._sel, ax, self.theta.value())
        self._result_atoms = _array_to_atoms(syms, coords)
        self.accept()

    def result_atoms(self):
        return self._result_atoms


# ═══════════════════════════════════════════════════════════
#  结构面板
# ═══════════════════════════════════════════════════════════

class StructurePanel(QWidget):
    """结构面板：坐标表格 + XYZ 文本编辑器 + 3D 预览，三者双向同步。"""
    atoms_changed = Signal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setSpacing(4)

        # ── 工具栏按钮行 ──────────────────────────
        row = QHBoxLayout(); row.setSpacing(4)
        self.btn_xyz  = QPushButton(t("btn_load_xyz"))
        self.btn_cif  = QPushButton(t("btn_load_cif"))
        self.btn_save_xyz = QPushButton(t("btn_save_xyz"))
        self.btn_3d   = QPushButton(t("btn_3d"))
        self.btn_add  = QPushButton(t("btn_add"))
        self.btn_del  = QPushButton(t("btn_del"))
        # 新增三个编辑按钮
        self.btn_preset     = QPushButton(t("btn_preset"))
        self.btn_place_ads  = QPushButton(t("btn_place_ads"))
        self.btn_constraint = QPushButton(t("btn_constraint"))
        self.btn_transform  = QPushButton(t("btn_transform"))
        for b in (self.btn_xyz,self.btn_cif,
                  self.btn_save_xyz,self.btn_3d,self.btn_add,self.btn_del,
                  self.btn_preset,self.btn_place_ads,self.btn_constraint,self.btn_transform):
            row.addWidget(b)
        row.addStretch(); lay.addLayout(row)

        # ── 提示标签 ──────────────────────────────
        self.lbl_hint = QLabel(t("lbl_struct_hint"))
        self.lbl_hint.setStyleSheet("color:#888888;font-size:8pt;"); self.lbl_hint.setWordWrap(True)
        lay.addWidget(self.lbl_hint)

        # ── 主体：左=内容Tab（表格/XYZ文本），右=3D视图 ──
        sp = QSplitter(Qt.Orientation.Horizontal)

        # 左侧：子 Tab（坐标表格 / XYZ 文本）
        self.left_tabs = QTabWidget()

        # 坐标表格 Tab
        tbl_w = QWidget(); tbl_lay = QVBoxLayout(tbl_w); tbl_lay.setContentsMargins(0,4,0,0)
        self.table = QTableWidget(0,4)
        self.table.setHorizontalHeaderLabels([t("col_elem"),t("col_x"),t("col_y"),t("col_z")])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setItemDelegate(ElementDelegate(self.table))
        tbl_lay.addWidget(self.table)
        self.left_tabs.addTab(tbl_w, t("tab_table"))

        # XYZ 文本编辑 Tab
        xyz_w = QWidget(); xyz_lay = QVBoxLayout(xyz_w); xyz_lay.setContentsMargins(0,4,0,0)
        self.xyz_edit = QPlainTextEdit()
        self.xyz_edit.setFont(QFont("Courier New", 10))
        self.xyz_edit.setPlaceholderText(t("xyz_placeholder"))
        self.xyz_edit.setStyleSheet(
            "QPlainTextEdit{background:#0f1724;color:#d8e6f3;"
            "font-family:'Courier New',monospace;font-size:10pt;"
            "border:1px solid #253244;border-radius:4px;}"
        )
        xyz_lay.addWidget(self.xyz_edit)
        # 应用按钮 + 行数提示
        xyz_btn_row = QHBoxLayout()
        self.btn_apply_xyz = QPushButton(t("btn_apply_xyz"))
        self.btn_apply_xyz.setFixedWidth(120)
        self.lbl_xyz_stat = QLabel("")
        self.lbl_xyz_stat.setStyleSheet("color:#888888;font-size:8pt;")
        xyz_btn_row.addWidget(self.btn_apply_xyz); xyz_btn_row.addWidget(self.lbl_xyz_stat,1)
        xyz_lay.addLayout(xyz_btn_row)
        self.left_tabs.addTab(xyz_w, t("tab_xyztext"))

        sp.addWidget(self.left_tabs)

        # 右侧:3D 视图(可拾取/可拖拽版)
        self.v3d = PickableStructure3DWidget(); self.v3d.setMinimumWidth(300)
        self.v3d.selection_changed.connect(self._on_3d_selection)
        self.v3d.atoms_dragged.connect(self._on_3d_dragged)
        sp.addWidget(self.v3d)
        sp.setStretchFactor(0,2); sp.setStretchFactor(1,3)
        lay.addWidget(sp, stretch=1)

        # 状态标签（显示文件名/原子数）
        self.lbl = QLabel(""); self.lbl.setStyleSheet("color:#88bbdd;font-size:8pt;")
        lay.addWidget(self.lbl)

        # 信号连接
        self.btn_xyz.clicked.connect(self._load_xyz)
        self.btn_cif.clicked.connect(self._load_cif)
        self.btn_save_xyz.clicked.connect(self._save_xyz)
        self.btn_3d.clicked.connect(self._view3d)
        self.btn_add.clicked.connect(self._add)
        self.btn_del.clicked.connect(self._del)
        self.table.itemChanged.connect(self._table_changed)
        self.btn_apply_xyz.clicked.connect(self._apply_xyz_text)
        self.xyz_edit.textChanged.connect(self._validate_xyz_live)
        # 切换到 XYZ 文本 Tab 时自动刷新文本内容
        self.left_tabs.currentChanged.connect(self._on_tab_change)
        # 新增三个编辑按钮
        self.btn_preset.clicked.connect(self._on_preset)
        self.btn_place_ads.clicked.connect(self._on_place_adsorbate)
        self.btn_constraint.clicked.connect(self._on_constraint)
        self.btn_transform.clicked.connect(self._on_transform)
        # 表格行选区 ↔ 3D 选区 双向同步
        self.table.itemSelectionChanged.connect(self._on_table_selection)
        # 更新更详细的提示文本
        self.lbl_hint.setText(t("lbl_struct_hint2"))
        # 启动时:空状态(显示 placeholder 而非示例分子)
        self.set_atoms([], t("struct_empty_hint"))

    # ── 原子数据同步 ─────────────────────────────

    def set_atoms(self, atoms, info=""):
        """主入口：设置原子列表，同步表格 + XYZ文本 + 3D视图。"""
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for sym,x,y,z in atoms:
            r = self.table.rowCount(); self.table.insertRow(r)
            self.table.setItem(r,0,QTableWidgetItem(sym))
            for c,v in enumerate([x,y,z],1):
                self.table.setItem(r,c,QTableWidgetItem(f"{v:.6f}"))
        self._validate_table(mark=True)
        self.table.blockSignals(False)
        # 同步 XYZ 文本区
        self._sync_xyz_text(atoms)
        self.lbl.setText(info)
        self.v3d.update_structure(atoms, reset_view=True)
        self.atoms_changed.emit(atoms)

    def _sync_xyz_text(self, atoms):
        """将 atoms 列表渲染为 .xyz 文本并写入编辑框（不触发解析信号）。"""
        n = len(atoms)
        lines = [str(n), f"{n} atoms"]
        for sym,x,y,z in atoms:
            lines.append(f"{sym:<4s}  {x:12.6f}  {y:12.6f}  {z:12.6f}")
        self.xyz_edit.blockSignals(True)
        self.xyz_edit.setPlainText("\n".join(lines))
        self.xyz_edit.blockSignals(False)
        self.lbl_xyz_stat.setText(f"{n} atoms")

    @Slot()
    def _table_changed(self):
        """表格被编辑 → 同步到 XYZ 文本 + 3D。"""
        ok, _ = self._validate_table(mark=True)
        if not ok:
            self.lbl.setText(t("err_table_live"))
            return
        a = self.get_atoms()
        self._sync_xyz_text(a)
        self.v3d.update_structure(a)
        self.atoms_changed.emit(a)

    @Slot(int)
    def _on_tab_change(self, idx):
        """切换到 XYZ 文本 Tab 时，刷新文本以反映表格最新内容。"""
        if idx == 1:
            self._sync_xyz_text(self.get_atoms())

    @Slot()
    def _apply_xyz_text(self):
        """解析 XYZ 文本 → 更新表格 + 3D。"""
        text = self.xyz_edit.toPlainText().strip()
        try:
            atoms = self._parse_xyz_text(text)
            self.table.blockSignals(True)
            self.table.setRowCount(0)
            for sym,x,y,z in atoms:
                r = self.table.rowCount(); self.table.insertRow(r)
                self.table.setItem(r,0,QTableWidgetItem(sym))
                for c,v in enumerate([x,y,z],1):
                    self.table.setItem(r,c,QTableWidgetItem(f"{v:.6f}"))
            self.table.blockSignals(False)
            self.lbl.setText(f"XYZ text — {len(atoms)} atoms")
            self.lbl_xyz_stat.setText(f"✓ {len(atoms)} atoms")
            self.v3d.update_structure(atoms)
            self.atoms_changed.emit(atoms)
        except Exception as e:
            self.lbl_xyz_stat.setText(f"✗ {e}")
            QMessageBox.warning(self, t("err_xyz_parse"), f"{t('err_xyz_detail')}\n{e}")

    @staticmethod
    def _parse_xyz_text(text: str) -> list:
        """解析 .xyz 格式文本，支持有/无行数头部两种格式。"""
        atoms, line_no, msg = StructurePanel._parse_xyz_text_detailed(text)
        if msg:
            raise ValueError(f"line {line_no or 1}: {msg}")
        return atoms

    @staticmethod
    def _parse_xyz_text_detailed(text: str) -> tuple[list, int | None, str | None]:
        """Parse XYZ and return atoms plus the first error line/message."""
        raw_lines = text.strip().splitlines()
        lines = [(i + 1, l.strip()) for i, l in enumerate(raw_lines) if l.strip()]
        if not lines:
            return [], 1, "Empty input"
        atoms = []
        # 判断是否有 XYZ 头
        start = 0
        try:
            n_declared = int(lines[0][1])
            start = 2  # 跳过行数行 + 注释行
        except ValueError:
            n_declared = None
            start = 0  # 无头部，直接解析坐标
        for line_no, line in lines[start:]:
            parts = line.split()
            if len(parts) >= 4:
                sym = parts[0].capitalize()
                if sym not in COMMON_ELEMENTS:
                    return atoms, line_no, f"Unknown element '{parts[0]}'"
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                except ValueError:
                    return atoms, line_no, "Coordinates must be numbers"
                atoms.append((sym, x, y, z))
            else:
                return atoms, line_no, "Expected: Element X Y Z"
        if not atoms:
            return atoms, 1, "No atom coordinates found"
        if n_declared is not None and n_declared != len(atoms):
            return atoms, 1, f"Declared {n_declared} atoms, parsed {len(atoms)}"
        return atoms, None, None

    @Slot()
    def _validate_xyz_live(self):
        text = self.xyz_edit.toPlainText().strip()
        if not text:
            self.lbl_xyz_stat.setText("")
            return
        atoms, line_no, msg = self._parse_xyz_text_detailed(text)
        if msg:
            self.lbl_xyz_stat.setStyleSheet("color:#f4a6b5;font-size:8pt;")
            self.lbl_xyz_stat.setText(t("err_xyz_live").format(line=line_no or 1, msg=msg))
        else:
            self.lbl_xyz_stat.setStyleSheet("color:#8ee6bc;font-size:8pt;")
            self.lbl_xyz_stat.setText(t("ok_xyz_live").format(n=len(atoms)))

    def _validate_table(self, mark=False):
        errors = []
        ok_bg = QColor("#111827")
        err_bg = QColor("#3a1d24")
        for r in range(self.table.rowCount()):
            elem_item = self.table.item(r, 0)
            sym = elem_item.text().strip().capitalize() if elem_item else ""
            elem_ok = sym in COMMON_ELEMENTS
            if not elem_ok:
                errors.append((r, 0))
            for c in range(1, 4):
                item = self.table.item(r, c)
                try:
                    float(item.text().strip()) if item else None
                    cell_ok = item is not None
                except Exception:
                    cell_ok = False
                if not cell_ok:
                    errors.append((r, c))
        if mark:
            for r in range(self.table.rowCount()):
                for c in range(4):
                    item = self.table.item(r, c)
                    if item:
                        item.setBackground(err_bg if (r, c) in errors else ok_bg)
        return not errors, errors

    # ── 导入 / 导出 ──────────────────────────────

    def _load_xyz(self):
        p,_ = QFileDialog.getOpenFileName(self, t("btn_load_xyz"), "", "XYZ (*.xyz);;All (*)")
        if not p: return
        try:
            raw = open(p, encoding="utf-8", errors="replace").read()
            atoms = self._parse_xyz_text(raw)
            self.set_atoms(atoms, f"XYZ: {Path(p).name}")
            # 保留原始文本（带注释）
            self.xyz_edit.blockSignals(True)
            self.xyz_edit.setPlainText(raw)
            self.xyz_edit.blockSignals(False)
            self.lbl_xyz_stat.setText(f"{len(atoms)} atoms  — {Path(p).name}")
        except Exception as e:
            QMessageBox.warning(self, t("err_xyz_parse"), f"{t('err_xyz_detail')}\n{e}")

    def _save_xyz(self):
        atoms = self.get_atoms()
        if not atoms:
            QMessageBox.warning(self, t("warn_title"), "No atoms to save."); return
        p,_ = QFileDialog.getSaveFileName(self, t("dlg_save_xyz"), "structure.xyz", "XYZ (*.xyz);;All (*)")
        if not p: return
        lines = [str(len(atoms)), "Exported from QED-DFT Studio"]
        for sym,x,y,z in atoms:
            lines.append(f"{sym:<4s}  {x:12.6f}  {y:12.6f}  {z:12.6f}")
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        self.lbl.setText(f"Saved: {Path(p).name}")

    def _load_cif(self):
        p,_ = QFileDialog.getOpenFileName(self, t("btn_load_cif"), "", "CIF (*.cif);;All (*)")
        if not p: return
        if HAS_GEMMI:
            try:
                doc=gemmi.cif.read(p); block=doc.sole_block()
                a,b,c=[float(block.find_value(f"_cell_length_{k}").split("(")[0]) for k in "abc"]
                al,be,ga=[float(block.find_value(f"_cell_angle_{k}").split("(")[0]) for k in ("alpha","beta","gamma")]
                cell=gemmi.UnitCell(a,b,c,al,be,ga); st=gemmi.make_small_structure_from_block(block)
                atoms=[]
                for s in st.sites:
                    pos=cell.orthogonalize(s.fract); sym=s.element.name if s.element else s.type_symbol
                    atoms.append((sym,pos.x,pos.y,pos.z))
                fm=(block.find_value("_chemical_formula_sum") or "?").strip()
                self.set_atoms(atoms,f"CIF: {Path(p).name} {fm}"); return
            except Exception as e: logger.warning(f"gemmi: {e}")
        if HAS_ASE:
            try:
                st=ase_read(p)
                atoms=[(s,float(x),float(y),float(z)) for s,(x,y,z) in zip(st.get_chemical_symbols(),st.get_positions())]
                self.set_atoms(atoms,f"CIF(ase): {Path(p).name}"); return
            except Exception as e: logger.warning(f"ase: {e}")
        QMessageBox.warning(self, t("err_cif_fail"), t("err_cif_install"))

    @Slot()
    def _view3d(self):
        atoms = self.get_atoms()
        if not atoms: return
        dlg = QDialog(self); dlg.setWindowTitle(t("dlg_3d_title")); dlg.resize(700,580)
        ll = QVBoxLayout(dlg)
        w = Structure3DWidget(); w.update_structure(atoms); ll.addWidget(w)
        b = QPushButton(t("dlg_3d_close")); b.clicked.connect(dlg.accept); ll.addWidget(b)
        dlg.exec()

    def _add(self):
        self.table.blockSignals(True)
        r = self.table.rowCount(); self.table.insertRow(r)
        self.table.setItem(r,0,QTableWidgetItem("C"))
        for c in range(1,4): self.table.setItem(r,c,QTableWidgetItem("0.000000"))
        self.table.blockSignals(False)
        self._table_changed()

    def _del(self):
        for r in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(r)
        self._table_changed()

    def get_atoms(self):
        atoms = []
        for r in range(self.table.rowCount()):
            try:
                sym = self.table.item(r,0).text().strip().capitalize()
                if sym not in COMMON_ELEMENTS:
                    continue
                x,y,z = [float(self.table.item(r,c).text()) for c in range(1,4)]
                atoms.append((sym,x,y,z))
            except: pass
        return atoms

    # ── 3D 视图编辑能力的 slot ──────────────────────────────

    @Slot(set)
    def _on_3d_selection(self, sel: set):
        """3D 视图选中原子的回调:同步表格行 + 状态行。"""
        self.table.blockSignals(True)
        self.table.clearSelection()
        for idx in sel:
            self.table.selectRow(idx)
        self.table.blockSignals(False)
        atoms = self.get_atoms()
        s = sorted(sel)
        msg = ""
        if len(s) == 1:
            i = s[0]; sym, x, y, z = atoms[i]
            msg = t("status_pick").format(i=i, sym=sym, x=x, y=y, z=z)
        elif len(s) == 2:
            i, j = s
            d = _bond_length(atoms[i][1:4], atoms[j][1:4])
            msg = t("status_dist").format(i=i, j=j, d=d)
        elif len(s) == 3:
            i, j, k = s
            a = _bond_angle(atoms[i][1:4], atoms[j][1:4], atoms[k][1:4])
            msg = t("status_angle").format(i=i, j=j, k=k, a=a)
        elif len(s) == 4:
            i, j, k, l = s
            a = _dihedral(atoms[i][1:4], atoms[j][1:4],
                          atoms[k][1:4], atoms[l][1:4])
            msg = t("status_dihedral").format(i=i, j=j, k=k, l=l, a=a)
        if msg:
            self.lbl.setText(msg)

    @Slot()
    def _on_table_selection(self):
        """表格行选区 → 3D 视图选区(避免与 3D→表格 形成循环)。"""
        rows = {i.row() for i in self.table.selectedIndexes()}
        if rows != self.v3d.get_selection():
            # set_selection 会发 selection_changed,_on_3d_selection 里的
            # blockSignals 已经能阻止表格再次触发本函数
            self.v3d.set_selection(rows)

    @Slot(list)
    def _on_3d_dragged(self, new_atoms):
        """右键拖动原子结束 → 同步回表格 / XYZ 文本。"""
        self.set_atoms(new_atoms, self.lbl.text() or "Dragged")

    @Slot()
    def _on_preset(self):
        """打开预设载入对话框。"""
        dlg = PresetLoaderDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            try:
                atoms = dlg.atoms()
                cl, ads = dlg.selection()
                label = f"Preset: {cl}" + (f"+{ads}" if ads else " (bare)")
                self.set_atoms(atoms, label)
            except Exception as e:
                QMessageBox.warning(self, t("dlg_preset_title"),
                                    f"{type(e).__name__}: {e}")

    @Slot()
    def _on_place_adsorbate(self):
        """Place an adsorbate on the selected atom or highest Cu atom."""
        atoms = self.get_atoms()
        if not atoms:
            QMessageBox.warning(self, t("dlg_ads_title"), t("err_no_ads_anchor"))
            return
        sel = self.v3d.get_selection()
        if not sel:
            sel = {i.row() for i in self.table.selectedIndexes()}
        dlg = AdsorbatePlacementDialog(atoms, sel, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.set_atoms(dlg.result_atoms(), "Adsorbate placed")

    @Slot()
    def _on_constraint(self):
        """打开几何约束对话框(基于 3D 选区或表格行选区)。"""
        atoms = self.get_atoms()
        if not atoms:
            return
        sel = self.v3d.get_selection()
        if not sel:
            sel = {i.row() for i in self.table.selectedIndexes()}
        dlg = GeomConstraintDialog(atoms, sel, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.set_atoms(dlg.result_atoms(), "Constraint applied")

    @Slot()
    def _on_transform(self):
        """打开整体平移旋转对话框(选区,空选区 → 全部)。"""
        atoms = self.get_atoms()
        if not atoms:
            return
        sel = self.v3d.get_selection()
        if not sel:
            sel = {i.row() for i in self.table.selectedIndexes()}
        dlg = TransformDialog(atoms, sel, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.set_atoms(dlg.result_atoms(), "Transform applied")

    def retranslate(self):
        """语言切换时刷新本面板所有文本。"""
        self.btn_xyz.setText(t("btn_load_xyz")); self.btn_cif.setText(t("btn_load_cif"))
        self.btn_save_xyz.setText(t("btn_save_xyz")); self.btn_3d.setText(t("btn_3d"))
        self.btn_add.setText(t("btn_add")); self.btn_del.setText(t("btn_del"))
        self.btn_apply_xyz.setText(t("btn_apply_xyz"))
        self.btn_preset.setText(t("btn_preset"))
        self.btn_place_ads.setText(t("btn_place_ads"))
        self.btn_constraint.setText(t("btn_constraint"))
        self.btn_transform.setText(t("btn_transform"))
        self.lbl_hint.setText(t("lbl_struct_hint2"))
        self.xyz_edit.setPlaceholderText(t("xyz_placeholder"))
        self.table.setHorizontalHeaderLabels([t("col_elem"),t("col_x"),t("col_y"),t("col_z")])
        self.left_tabs.setTabText(0, t("tab_table"))
        self.left_tabs.setTabText(1, t("tab_xyztext"))


# ═══════════════════════════════════════════════════════════
#  计算设置面板（整合通用 + nQEDDFT + 流水线）
# ═══════════════════════════════════════════════════════════

class CalcSettingsPanel(QWidget):
    load_cluster = Signal(list, str)   # → 结构面板

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QHBoxLayout(self); root.setSpacing(6)

        # ── 左：参数滚动区 ────────────────────────
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(370); scroll.setMaximumWidth(450)
        pw = QWidget(); pl = QVBoxLayout(pw); pl.setSpacing(8)
        scroll.setWidget(pw); root.addWidget(scroll)

        # ── § 1 通用计算方法 ──────────────────────
        self.grp_method = QGroupBox(t("grp_method")); fm = QFormLayout(self.grp_method); fm.setSpacing(5)
        self.combo_method = QComboBox(); self.combo_method.addItems(["HF","DFT","QED-DFT"])
        self.rl_method  = QLabel(t("lbl_method"));  fm.addRow(self.rl_method,  self.combo_method)
        self.combo_xc = QComboBox()
        self.combo_xc.addItems(["pbe0","b3lyp","pbe","tpss","m06-2x","wb97x-d"])
        self.rl_xc      = QLabel(t("lbl_xc"));      fm.addRow(self.rl_xc,      self.combo_xc)
        self.combo_basis = QComboBox()
        self.combo_basis.addItems(["def2-SVP","def2-TZVP","6-31G*","6-31G**","6-311G**","cc-pVDZ","cc-pVTZ"])
        self.rl_basis   = QLabel(t("lbl_basis"));   fm.addRow(self.rl_basis,   self.combo_basis)
        self.spin_charge = QSpinBox(); self.spin_charge.setRange(-10,10)
        self.rl_charge  = QLabel(t("lbl_charge"));  fm.addRow(self.rl_charge,  self.spin_charge)
        self.spin_mult = QSpinBox(); self.spin_mult.setRange(1,10); self.spin_mult.setValue(1)
        self.rl_mult    = QLabel(t("lbl_mult"));    fm.addRow(self.rl_mult,    self.spin_mult)
        self.lbl_auto_spin = QLabel(t("lbl_auto_spin"))
        self.lbl_auto_spin.setStyleSheet("color:#aaaaaa;font-size:8pt;")
        fm.addRow("", self.lbl_auto_spin); pl.addWidget(self.grp_method)

        # ── § 2 SCF 参数 ──────────────────────────
        self.grp_scf = QGroupBox(t("grp_scf")); fs = QFormLayout(self.grp_scf); fs.setSpacing(5)
        self.spin_maxcyc = QSpinBox(); self.spin_maxcyc.setRange(10,2000); self.spin_maxcyc.setValue(300)
        self.rl_maxcyc = QLabel(t("lbl_maxcyc")); fs.addRow(self.rl_maxcyc, self.spin_maxcyc)
        self.dspin_tol = QDoubleSpinBox(); self.dspin_tol.setDecimals(12)
        self.dspin_tol.setRange(1e-15,1e-3); self.dspin_tol.setValue(1e-9); self.dspin_tol.setSingleStep(1e-10)
        self.rl_tol    = QLabel(t("lbl_tol"));    fs.addRow(self.rl_tol,    self.dspin_tol)
        self.spin_diis = QSpinBox(); self.spin_diis.setRange(2,20); self.spin_diis.setValue(8)
        self.rl_diis   = QLabel(t("lbl_diis"));   fs.addRow(self.rl_diis,   self.spin_diis)
        self.dspin_ls = QDoubleSpinBox(); self.dspin_ls.setRange(0,1); self.dspin_ls.setValue(0.2); self.dspin_ls.setDecimals(2)
        self.rl_ls     = QLabel(t("lbl_ls"));     fs.addRow(self.rl_ls,     self.dspin_ls)

        self.grp_advanced = QGroupBox(t("grp_advanced"))
        self.grp_advanced.setCheckable(True)
        self.grp_advanced.setChecked(False)
        adv_outer = QVBoxLayout(self.grp_advanced)
        adv_outer.setSpacing(6)
        self.lbl_advanced_hint = QLabel(t("advanced_hint"))
        self.lbl_advanced_hint.setWordWrap(True)
        self.lbl_advanced_hint.setStyleSheet("color:#8fa0b3;font-size:8pt;")
        adv_outer.addWidget(self.lbl_advanced_hint)
        self.adv_body = QWidget()
        self.adv_lay = QVBoxLayout(self.adv_body)
        self.adv_lay.setContentsMargins(0, 0, 0, 0)
        self.adv_lay.setSpacing(8)
        self.adv_lay.addWidget(self.grp_scf)
        adv_outer.addWidget(self.adv_body)
        self.adv_body.setVisible(False)
        self.grp_advanced.toggled.connect(self.adv_body.setVisible)
        pl.addWidget(self.grp_advanced)

        # ── § 3 nQEDDFT 体系 ──────────────────────
        self.grp_sys = QGroupBox(t("grp_sys"))
        self.grp_sys.setCheckable(True); self.grp_sys.setChecked(False)
        fsy = QFormLayout(self.grp_sys); fsy.setSpacing(5)
        self.combo_cluster = QComboBox(); self.combo_cluster.addItems(list(CLUSTERS.keys()))
        self.rl_cluster = QLabel(t("lbl_cluster")); fsy.addRow(self.rl_cluster, self.combo_cluster)
        self.lbl_cl = QLabel(); self.lbl_cl.setStyleSheet("color:#7ec8e3;font-size:8pt;")
        fsy.addRow("", self.lbl_cl)
        self.combo_ads = QComboBox(); self.combo_ads.addItem(t("ads_none"))
        self.combo_ads.addItems(list(ADS_OFFSETS.keys()))
        self.rl_ads     = QLabel(t("lbl_ads"));     fsy.addRow(self.rl_ads,     self.combo_ads)
        self.lbl_ads_info = QLabel(); self.lbl_ads_info.setStyleSheet("color:#7ec8e3;font-size:8pt;")
        fsy.addRow("", self.lbl_ads_info)
        self.combo_gasref = QComboBox(); self.combo_gasref.addItem(t("gasref_auto"))
        self.combo_gasref.addItems(list(GAS_REFS.keys()))
        self.rl_gasref  = QLabel(t("lbl_gasref"));  fsy.addRow(self.rl_gasref,  self.combo_gasref)
        self.btn_sync = QPushButton(t("btn_sync_geom"))
        self.btn_sync.clicked.connect(self._sync_geom); fsy.addRow("", self.btn_sync)
        pl.addWidget(self.grp_sys)

        # ── § 4 腔场参数 ──────────────────────────
        self.grp_cav = QGroupBox(t("grp_cav"))
        self.grp_cav.setCheckable(True); self.grp_cav.setChecked(False)
        fcv = QFormLayout(self.grp_cav); fcv.setSpacing(5)
        self.dspin_omega = QDoubleSpinBox(); self.dspin_omega.setRange(0.001,5.0)
        self.dspin_omega.setValue(0.0856); self.dspin_omega.setDecimals(5); self.dspin_omega.setSingleStep(0.005)
        self.dspin_omega.setToolTip("0.0856 a.u. ≈ 1880 cm⁻¹")
        self.rl_omega  = QLabel(t("lbl_omega"));  fcv.addRow(self.rl_omega,  self.dspin_omega)
        self.dspin_lam = QDoubleSpinBox(); self.dspin_lam.setRange(0,0.5)
        self.dspin_lam.setValue(0.02); self.dspin_lam.setDecimals(4); self.dspin_lam.setSingleStep(0.005)
        self.rl_lambda = QLabel(t("lbl_lambda")); fcv.addRow(self.rl_lambda, self.dspin_lam)
        self.combo_pol = QComboBox()
        # 存储极化索引 0=auto 1=z 2=x 3=y，文本随语言变化
        self.combo_pol.addItems([t("pol_auto"), t("pol_z"), t("pol_x"), t("pol_y")])
        self.rl_pol    = QLabel(t("lbl_pol"));    fcv.addRow(self.rl_pol,    self.combo_pol); pl.addWidget(self.grp_cav)

        # ── § 5 连续任务流水线 ────────────────────
        self.grp_pipe = QGroupBox(t("grp_pipe"))
        fpipe = QVBoxLayout(self.grp_pipe); fpipe.setSpacing(3)
        self.lbl_pipe_hint = QLabel(t("pipe_hint"))
        self.lbl_pipe_hint.setStyleSheet("color:#aaaaaa;font-size:8pt;"); self.lbl_pipe_hint.setWordWrap(True)
        fpipe.addWidget(self.lbl_pipe_hint)

        self.chk_geom_opt  = self._tchk(t("chk_geom"))
        self.chk_gas_ref   = self._tchk(t("chk_gas"))
        self.chk_freq0     = self._tchk(t("chk_freq0"))
        self.chk_scan      = self._tchk(t("chk_scan"))
        self.chk_freq_cav  = self._tchk(t("chk_freqcav"))
        self.chk_polariton = self._tchk(t("chk_pol"))
        for c in (self.chk_geom_opt,self.chk_gas_ref,self.chk_freq0,
                  self.chk_scan,self.chk_freq_cav,self.chk_polariton):
            fpipe.addWidget(c)

        # λ 扫描子参数
        self.grp_scan_sub = QGroupBox(t("grp_scan_sub")); self.grp_scan_sub.setVisible(False)
        fss = QFormLayout(self.grp_scan_sub); fss.setSpacing(4)
        self.chk_preset = QCheckBox(t("chk_preset"))
        self.chk_preset.setChecked(True); fss.addRow(self.chk_preset)
        self.edit_lams = QLineEdit("0.0, 0.01, 0.02, 0.05")
        self.edit_lams.setEnabled(False)
        self.rl_custom_lam = QLabel(t("lbl_custom_lambda")); fss.addRow(self.rl_custom_lam, self.edit_lams)
        self.chk_preset.stateChanged.connect(lambda s: self.edit_lams.setEnabled(not bool(s)))
        fpipe.addWidget(self.grp_scan_sub)
        self.chk_scan.stateChanged.connect(lambda s: self.grp_scan_sub.setVisible(bool(s)))
        pl.addWidget(self.grp_pipe)

        # ── § 5b Stage 流水线(独立子进程驱动 stage_pipeline.py)──
        self.grp_stages = QGroupBox(t("grp_stages"))
        self.grp_stages.setCheckable(True); self.grp_stages.setChecked(False)
        fst = QVBoxLayout(self.grp_stages); fst.setSpacing(3)
        self.lbl_stages_hint = QLabel(t("stages_hint"))
        self.lbl_stages_hint.setStyleSheet("color:#aaaaaa;font-size:8pt;")
        self.lbl_stages_hint.setWordWrap(True)
        fst.addWidget(self.lbl_stages_hint)

        self.chk_stage0 = self._tchk(t("chk_stage0")); self.chk_stage0.setChecked(True)
        self.chk_stage1 = self._tchk(t("chk_stage1"))
        self.chk_stage2 = self._tchk(t("chk_stage2"))
        for c in (self.chk_stage0, self.chk_stage1, self.chk_stage2):
            fst.addWidget(c)

        # workdir + script path
        srow = QFormLayout(); srow.setSpacing(3)
        self.edit_stage_workdir = QLineEdit("./run")
        self.rl_stage_workdir   = QLabel(t("lbl_stage_workdir"))
        srow.addRow(self.rl_stage_workdir, self.edit_stage_workdir)
        self.edit_stage_script  = QLineEdit("stage_pipeline.py")
        self.rl_stage_script    = QLabel(t("lbl_stage_script"))
        srow.addRow(self.rl_stage_script, self.edit_stage_script)
        fst.addLayout(srow)
        self.adv_lay.addWidget(self.grp_stages)

        # 当 Stage 流水线启用时,自动禁用旧的细粒度流水线(避免冲突)
        self.grp_stages.toggled.connect(
            lambda on: self.grp_pipe.setEnabled(not on))

        # ── § 6 断点续算 ──────────────────────────
        self.grp_ckpt = QGroupBox(t("grp_ckpt")); fck = QFormLayout(self.grp_ckpt)
        self.edit_ckpt = QLineEdit("nqed_ckpt.json")
        self.rl_ckpt   = QLabel(t("lbl_checkpoint")); fck.addRow(self.rl_ckpt, self.edit_ckpt)
        self.chk_resume = QCheckBox(t("chk_resume")); fck.addRow(self.chk_resume)
        self.adv_lay.addWidget(self.grp_ckpt)
        pl.addStretch()

        # ── 右：预览区 ───────────────────────────
        rw = QWidget(); rl = QVBoxLayout(rw); rl.setContentsMargins(0,0,0,0); rl.setSpacing(6)

        self.lbl_preview_cluster = QLabel(t("lbl_preview_cluster_title"))
        rl.addWidget(self.lbl_preview_cluster,0)
        self.cluster_v3d = Structure3DWidget(); self.cluster_v3d.setMinimumHeight(220)
        rl.addWidget(self.cluster_v3d,2)
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine); rl.addWidget(sep)
        self.lbl_preview_scan = QLabel(t("lbl_preview_scan_title"))
        rl.addWidget(self.lbl_preview_scan,0)
        self.scan_prev = LambdaPlot(); rl.addWidget(self.scan_prev,3)
        root.addWidget(rw,stretch=1)

        # 信号
        self.combo_cluster.currentTextChanged.connect(self._upd_cluster)
        self.combo_ads.currentTextChanged.connect(self._upd_ads)
        self.grp_sys.toggled.connect(self._sys_toggled)
        self.combo_method.currentTextChanged.connect(self._method_changed)
        self._upd_cluster(self.combo_cluster.currentText())
        self._upd_ads(self.combo_ads.currentText())

    @staticmethod
    def _tchk(label):
        c = QCheckBox(label); c.setChecked(True)
        c.setStyleSheet("QCheckBox{color:#ffffff;padding:2px 0;}"); return c

    def _upd_cluster(self, name):
        if name not in CLUSTERS: return
        cl = CLUSTERS[name]; n = len([l for l in cl["geom"].strip().split("\n")])
        st = "doublet/UKS" if cl["spin"] else "singlet/RKS"
        self.lbl_cl.setText(t("lbl_cluster_atoms").format(n=n, st=st))
        ads_name = self.combo_ads.currentText() if hasattr(self,"combo_ads") else None
        ads = ads_name if ads_name in ADS_OFFSETS else None
        self.cluster_v3d.update_structure(parse_geom(build_geom(name, ads)))

    def _upd_ads(self, name):
        if name in ADS_OFFSETS:
            a = ADS_OFFSETS[name]
            self.lbl_ads_info.setText(t("lbl_ads_info").format(ne=a["n_electrons"], nu=a["nu_target_cm"]))
        else:
            self.lbl_ads_info.setText("")
        cluster = self.combo_cluster.currentText()
        ads = name if name in ADS_OFFSETS else None
        self.cluster_v3d.update_structure(parse_geom(build_geom(cluster, ads)))

    def _sys_toggled(self, checked):
        self.grp_cav.setChecked(checked)
        if checked: self.combo_method.setCurrentText("QED-DFT")

    def _method_changed(self, method):
        is_qed = (method == "QED-DFT")
        self.grp_sys.setChecked(is_qed); self.grp_cav.setChecked(is_qed)
        self.combo_xc.setEnabled(method != "HF")

    @Slot()
    def _sync_geom(self):
        cluster = self.combo_cluster.currentText()
        ads_raw = self.combo_ads.currentText()
        ads = ads_raw if ads_raw in ADS_OFFSETS else None
        atoms = parse_geom(build_geom(cluster, ads))
        label = f"nQEDDFT: {cluster}" + (f" + {ads}" if ads else "")
        self.load_cluster.emit(atoms, label)

    def get_pol(self):
        idx = self.combo_pol.currentIndex()
        if idx == 2: return [1.,0.,0.]   # x
        if idx == 3: return [0.,1.,0.]   # y
        if idx == 1: return [0.,0.,1.]   # z
        return [0.,0.,1.]                # auto → z

    def get_lams(self):
        if self.chk_preset.isChecked(): return list(LAMBDA_PRESETS)
        try: return [float(x.strip()) for x in self.edit_lams.text().split(",")]
        except: return list(LAMBDA_PRESETS)

    def get_config(self) -> dict:
        cluster = self.combo_cluster.currentText()
        ads_raw = self.combo_ads.currentText()
        ads = ads_raw if ads_raw in ADS_OFFSETS else None
        use_nqed = self.grp_sys.isChecked() and HAS_NQEDDFT
        spin = calc_spin(cluster, ads) if use_nqed else self.spin_mult.value()-1
        gasref = self.combo_gasref.currentText()
        # "自动推断" (zh) or "Auto" (en) – use TR values for both langs
        _auto_vals = {TR["gasref_auto"]["zh"], TR["gasref_auto"]["en"]}
        if gasref in _auto_vals or gasref not in GAS_REFS: gasref = GAS_MAP.get(ads) if ads else None
        return {
            "method": self.combo_method.currentText(),
            "xc": self.combo_xc.currentText(),
            "basis": self.combo_basis.currentText(),
            "charge": self.spin_charge.value(),
            "spin": spin,
            "max_cycle": self.spin_maxcyc.value(),
            "conv_tol": self.dspin_tol.value(),
            "diis_space": self.spin_diis.value(),
            "level_shift": self.dspin_ls.value(),
            "use_nqed": use_nqed,
            "cluster": cluster, "ads": ads, "gas_ref": gasref,
            "use_cavity": self.grp_cav.isChecked(),
            "omega_c": self.dspin_omega.value(),
            "lambda_val": self.dspin_lam.value(),
            "polarization": self.get_pol(),
            "do_geom_opt":  self.chk_geom_opt.isChecked(),
            "do_gas_ref":   self.chk_gas_ref.isChecked(),
            "do_freq0":     self.chk_freq0.isChecked(),
            "do_scan":      self.chk_scan.isChecked(),
            "do_freq_cav":  self.chk_freq_cav.isChecked(),
            "do_polariton": self.chk_polariton.isChecked(),
            "lambda_list":  self.get_lams(),
            "checkpoint":   self.edit_ckpt.text(),
            "resume":       self.chk_resume.isChecked(),
            # ── Stage 流水线(stage_pipeline.py 子进程驱动)──
            "use_stages":   self.grp_stages.isChecked(),
            "stages":       [s for s, chk in [
                                (0, self.chk_stage0),
                                (1, self.chk_stage1),
                                (2, self.chk_stage2),
                            ] if chk.isChecked()],
            "stage_workdir": self.edit_stage_workdir.text(),
            "stage_script":  self.edit_stage_script.text(),
        }

    def set_config(self, cfg: dict) -> None:
        """从 dict 恢复设置(用于打开项目时回填)。未提供的键保留当前值。"""
        if not cfg:
            return
        # 文本下拉框: 用 setCurrentText (找不到时静默忽略)
        def _setcb(combo, key):
            val = cfg.get(key)
            if val is None: return
            idx = combo.findText(str(val))
            if idx >= 0: combo.setCurrentIndex(idx)
        _setcb(self.combo_method, "method")
        _setcb(self.combo_xc,     "xc")
        _setcb(self.combo_basis,  "basis")
        _setcb(self.combo_cluster,"cluster")

        # ads: None → 选 "—" 或第一项
        ads = cfg.get("ads")
        if ads is None:
            self.combo_ads.setCurrentIndex(0)
        else:
            idx = self.combo_ads.findText(str(ads))
            if idx >= 0: self.combo_ads.setCurrentIndex(idx)

        # gas_ref: None → 选 "自动推断"
        gas = cfg.get("gas_ref")
        if gas is None:
            for k in (TR["gasref_auto"]["zh"], TR["gasref_auto"]["en"]):
                idx = self.combo_gasref.findText(k)
                if idx >= 0:
                    self.combo_gasref.setCurrentIndex(idx); break
        else:
            idx = self.combo_gasref.findText(str(gas))
            if idx >= 0: self.combo_gasref.setCurrentIndex(idx)

        # 数值字段
        for spin, key in [
            (self.spin_charge,  "charge"),
            (self.spin_mult,    "spin"),     # 注意:get_config 返回 spin 而不是 mult
            (self.spin_maxcyc,  "max_cycle"),
            (self.spin_diis,    "diis_space"),
        ]:
            if key in cfg and cfg[key] is not None:
                # spin → mult: mult = spin + 1
                v = cfg[key] + 1 if key == "spin" else cfg[key]
                spin.setValue(int(v))
        for dspin, key in [
            (self.dspin_tol,   "conv_tol"),
            (self.dspin_ls,    "level_shift"),
            (self.dspin_omega, "omega_c"),
            (self.dspin_lam,   "lambda_val"),
        ]:
            if key in cfg and cfg[key] is not None:
                dspin.setValue(float(cfg[key]))

        # GroupBox checkable
        if "use_nqed"   in cfg: self.grp_sys.setChecked(bool(cfg["use_nqed"]))
        if "use_cavity" in cfg: self.grp_cav.setChecked(bool(cfg["use_cavity"]))

        # 流水线复选框
        for chk, key in [
            (self.chk_geom_opt,  "do_geom_opt"),
            (self.chk_gas_ref,   "do_gas_ref"),
            (self.chk_freq0,     "do_freq0"),
            (self.chk_scan,      "do_scan"),
            (self.chk_freq_cav,  "do_freq_cav"),
            (self.chk_polariton, "do_polariton"),
            (self.chk_resume,    "resume"),
        ]:
            if key in cfg and cfg[key] is not None:
                chk.setChecked(bool(cfg[key]))

        # λ 列表 + 断点路径
        lams = cfg.get("lambda_list")
        if lams and hasattr(self, "edit_lams"):
            try:
                self.edit_lams.setText(",".join(f"{float(x):g}" for x in lams))
            except Exception:
                pass
        ckpt = cfg.get("checkpoint")
        if ckpt is not None:
            self.edit_ckpt.setText(str(ckpt))

        # ── Stage 流水线 ──────────────────────────────
        if "use_stages" in cfg:
            self.grp_stages.setChecked(bool(cfg["use_stages"]))
        stages = cfg.get("stages")
        if isinstance(stages, list):
            self.chk_stage0.setChecked(0 in stages)
            self.chk_stage1.setChecked(1 in stages)
            self.chk_stage2.setChecked(2 in stages)
        if cfg.get("stage_workdir"):
            self.edit_stage_workdir.setText(str(cfg["stage_workdir"]))
        if cfg.get("stage_script"):
            self.edit_stage_script.setText(str(cfg["stage_script"]))
        if cfg.get("use_stages") or cfg.get("resume"):
            self.grp_advanced.setChecked(True)


    def retranslate(self):
        """语言切换时刷新计算设置面板所有文本。"""
        # GroupBox titles
        self.grp_method.setTitle(t("grp_method"))
        self.grp_scf.setTitle(t("grp_scf"))
        self.grp_advanced.setTitle(t("grp_advanced"))
        self.grp_sys.setTitle(t("grp_sys"))
        self.grp_cav.setTitle(t("grp_cav"))
        self.grp_pipe.setTitle(t("grp_pipe"))
        self.grp_scan_sub.setTitle(t("grp_scan_sub"))
        self.grp_ckpt.setTitle(t("grp_ckpt"))
        # Row-label widgets (stored as self.rl_* in __init__)
        self.rl_method.setText(t("lbl_method"))
        self.rl_xc.setText(t("lbl_xc"))
        self.rl_basis.setText(t("lbl_basis"))
        self.rl_charge.setText(t("lbl_charge"))
        self.rl_mult.setText(t("lbl_mult"))
        self.rl_maxcyc.setText(t("lbl_maxcyc"))
        self.rl_tol.setText(t("lbl_tol"))
        self.rl_diis.setText(t("lbl_diis"))
        self.rl_ls.setText(t("lbl_ls"))
        self.rl_cluster.setText(t("lbl_cluster"))
        self.rl_ads.setText(t("lbl_ads"))
        self.rl_gasref.setText(t("lbl_gasref"))
        self.rl_omega.setText(t("lbl_omega"))
        self.rl_lambda.setText(t("lbl_lambda"))
        self.rl_pol.setText(t("lbl_pol"))
        self.rl_ckpt.setText(t("lbl_checkpoint"))
        self.rl_custom_lam.setText(t("lbl_custom_lambda"))
        # Other labels & buttons
        self.lbl_auto_spin.setText(t("lbl_auto_spin"))
        self.lbl_advanced_hint.setText(t("advanced_hint"))
        self.lbl_pipe_hint.setText(t("pipe_hint"))
        self.btn_sync.setText(t("btn_sync_geom"))
        self.lbl_preview_cluster.setText(t("lbl_preview_cluster_title"))
        self.lbl_preview_scan.setText(t("lbl_preview_scan_title"))
        # Checkboxes
        self.chk_geom_opt.setText(t("chk_geom")); self.chk_gas_ref.setText(t("chk_gas"))
        self.chk_freq0.setText(t("chk_freq0")); self.chk_scan.setText(t("chk_scan"))
        self.chk_freq_cav.setText(t("chk_freqcav")); self.chk_polariton.setText(t("chk_pol"))
        self.chk_preset.setText(t("chk_preset")); self.chk_resume.setText(t("chk_resume"))
        # Stage pipeline 翻译
        self.grp_stages.setTitle(t("grp_stages"))
        self.lbl_stages_hint.setText(t("stages_hint"))
        self.chk_stage0.setText(t("chk_stage0"))
        self.chk_stage1.setText(t("chk_stage1"))
        self.chk_stage2.setText(t("chk_stage2"))
        self.rl_stage_workdir.setText(t("lbl_stage_workdir"))
        self.rl_stage_script.setText(t("lbl_stage_script"))
        # Polarization combobox (re-populate preserving index)
        idx = self.combo_pol.currentIndex()
        self.combo_pol.blockSignals(True)
        self.combo_pol.clear()
        self.combo_pol.addItems([t("pol_auto"), t("pol_z"), t("pol_x"), t("pol_y")])
        self.combo_pol.setCurrentIndex(max(0, idx))
        self.combo_pol.blockSignals(False)
        # Combobox first-item strings
        self.combo_ads.blockSignals(True)
        self.combo_ads.setItemText(0, t("ads_none"))
        self.combo_ads.blockSignals(False)
        self.combo_gasref.blockSignals(True)
        self.combo_gasref.setItemText(0, t("gasref_auto"))
        self.combo_gasref.blockSignals(False)
        # Refresh dynamic hint labels
        self._upd_cluster(self.combo_cluster.currentText())
        self._upd_ads(self.combo_ads.currentText())


# ═══════════════════════════════════════════════════════════
#  任务监控面板
# ═══════════════════════════════════════════════════════════

class MonitorPanel(QWidget):
    _STEP_KEYS = ["step_geom","step_gas","step_freq0","step_scan","step_freqcav","step_pol"]
    _STATE_KEYS = {
        "idle": "state_idle",
        "start": "state_running",
        "running": "state_running",
        "done": "state_done",
        "skip": "state_skip",
        "error": "state_error",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self); lay.setContentsMargins(4,4,4,4); lay.setSpacing(5)

        self.grp_pipeline = QGroupBox(t("grp_pipeline_status")); pipe_lay = QHBoxLayout(self.grp_pipeline); pipe_lay.setSpacing(4)
        self._step_lbls: list[QLabel] = []
        for key in self._STEP_KEYS:
            lbl = QLabel(t(key)); lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(self._sty("idle")); lbl.setFixedHeight(28)
            lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            pipe_lay.addWidget(lbl); self._step_lbls.append(lbl)
        lay.addWidget(self.grp_pipeline)

        prow = QHBoxLayout()
        self.lbl_step = QLabel(t("status_ready")); self.lbl_step.setStyleSheet("color:#ffffff;font-weight:bold;")
        self.prog = QProgressBar(); self.prog.setRange(0,100); self.prog.setFixedHeight(16)
        prow.addWidget(self.lbl_step,0); prow.addWidget(self.prog,1); lay.addLayout(prow)

        sp = QSplitter(Qt.Orientation.Vertical)
        self.grp_tasks = QGroupBox(t("grp_task_queue"))
        task_lay = QVBoxLayout(self.grp_tasks)
        task_lay.setContentsMargins(6, 6, 6, 6)
        self.task_table = QTableWidget(0, 5)
        self.task_table.setHorizontalHeaderLabels([
            t("col_task"), t("col_state"), t("col_started"),
            t("col_elapsed"), t("col_workdir")
        ])
        self.task_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.task_table.setAlternatingRowColors(True)
        self.task_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        task_lay.addWidget(self.task_table)
        sp.addWidget(self.grp_tasks)
        self.scf_plot = SCFPlotWidget(); self.scf_plot.setMinimumHeight(160); sp.addWidget(self.scf_plot)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setFont(QFont("Courier New",9))
        self.log.setMinimumHeight(80)
        self.log.setPlaceholderText(t("log_placeholder"))
        sp.addWidget(self.log)
        sp.setStretchFactor(0,1); sp.setStretchFactor(1,2); sp.setStretchFactor(2,3)
        lay.addWidget(sp,stretch=1)
        self._task_rows: dict[int, int] = {}
        self._task_started: dict[int, float] = {}

    @staticmethod
    def _sty(state):
        return {"idle":    "background:#1a1a1a;color:#666666;border-radius:4px;padding:2px 4px;font-size:9pt;",
                "running": "background:#2563eb;color:#fff;border-radius:4px;padding:2px 4px;font-size:9pt;font-weight:bold;",
                "done":    "background:#50fa7b;color:#1a1a2e;border-radius:4px;padding:2px 4px;font-size:9pt;font-weight:bold;",
                "skip":    "background:#0f0f0f;color:#444444;border-radius:4px;padding:2px 4px;font-size:9pt;",
                "error":   "background:#ff5555;color:#fff;border-radius:4px;padding:2px 4px;font-size:9pt;font-weight:bold;",
                }.get(state,"")

    def set_step(self, idx, state):
        if 0 <= idx < len(self._step_lbls): self._step_lbls[idx].setStyleSheet(self._sty(state))

    def reset(self):
        for l in self._step_lbls: l.setStyleSheet(self._sty("idle"))
        self.prog.setValue(0); self.lbl_step.setText(t("status_ready"))
        self.scf_plot.reset(); self.log.clear()
        self.task_table.setRowCount(0)
        self._task_rows.clear()
        self._task_started.clear()

    def seed_tasks(self, names: list[str], workdir: str = ""):
        self.task_table.setRowCount(0)
        self._task_rows.clear()
        self._task_started.clear()
        for idx, name in enumerate(names):
            self.update_task(idx, name, "idle", workdir)

    @Slot(dict)
    def update_task_event(self, event: dict):
        self.update_task(
            int(event.get("idx", 0)),
            str(event.get("name", "")),
            str(event.get("state", "running")),
            str(event.get("workdir", "")),
        )

    def update_task(self, idx: int, name: str, state: str, workdir: str = ""):
        if idx not in self._task_rows:
            row = self.task_table.rowCount()
            self.task_table.insertRow(row)
            self._task_rows[idx] = row
        row = self._task_rows[idx]
        state_key = self._STATE_KEYS.get(state, "state_running")
        if state in ("start", "running") and idx not in self._task_started:
            self._task_started[idx] = time.time()
        started = ""
        elapsed = ""
        if idx in self._task_started:
            started = time.strftime("%H:%M:%S", time.localtime(self._task_started[idx]))
            elapsed = f"{time.time() - self._task_started[idx]:.1f}s"
        vals = [name or f"stage{idx}", t(state_key), started, elapsed, workdir]
        for c, v in enumerate(vals):
            item = self.task_table.item(row, c)
            if item is None:
                item = QTableWidgetItem()
                self.task_table.setItem(row, c, item)
            item.setText(str(v))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter if c < 4 else Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            if c == 1:
                item.setForeground(QColor({
                    "done": "#8ee6bc", "skip": "#8fa0b3",
                    "error": "#f4a6b5", "start": "#93c5fd",
                    "running": "#93c5fd", "idle": "#8fa0b3",
                }.get(state, "#f3f4f6")))
        self.task_table.scrollToBottom()

    @Slot(str)
    def append_log(self, s): self.log.appendPlainText(s)
    @Slot(int)
    def set_prog(self, v):   self.prog.setValue(v)
    @Slot(str)
    def set_label(self, s):  self.lbl_step.setText(s)

    def retranslate(self):
        for lbl,key in zip(self._step_lbls, self._STEP_KEYS):
            lbl.setText(t(key))
        self.grp_pipeline.setTitle(t("grp_pipeline_status"))
        self.grp_tasks.setTitle(t("grp_task_queue"))
        self.task_table.setHorizontalHeaderLabels([
            t("col_task"), t("col_state"), t("col_started"),
            t("col_elapsed"), t("col_workdir")
        ])
        self.lbl_step.setText(t("status_ready"))
        self.log.setPlaceholderText(t("log_placeholder"))


# ═══════════════════════════════════════════════════════════
#  结果面板
# ═══════════════════════════════════════════════════════════

class ResultsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)

        # ── 卡片行 ───────────────────────────────
        crow = QHBoxLayout(); crow.setSpacing(8)
        self.c_energy  = self._card(t("card_energy"),    "—", "#2563eb")
        self.c_ads     = self._card(t("card_ads"),        "—", "#8b5cf6")
        self.c_freq0   = self._card(t("card_freq0"),     "— cm⁻¹", "#059669")
        self.c_dfreq   = self._card(t("card_dfreq"),     "— cm⁻¹", "#d97706")
        self.c_conv    = self._card(t("card_conv"),        "—", "#888888")
        for w in (self.c_energy,self.c_ads,self.c_freq0,self.c_dfreq,self.c_conv):
            crow.addWidget(w,1)
        lay.addLayout(crow)

        # ── 主体分割 ─────────────────────────────
        sp = QSplitter(Qt.Orientation.Horizontal)

        self.ltabs = QTabWidget()
        ltabs = self.ltabs
        # 振动表
        self.t_freq = QTableWidget(0,4)
        self.t_freq.setHorizontalHeaderLabels([t("col_mode"),t("col_freq"),t("col_intens"),t("col_annot")])
        self.t_freq.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.t_freq.setAlternatingRowColors(True); self.t_freq.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        ltabs.addTab(self.t_freq, t("tab_vib"))
        # λ 扫描表
        self.t_scan = QTableWidget(0,6)
        self.t_scan.setHorizontalHeaderLabels([t("col_lam"),t("col_e_ha"),t("col_de_ads"),t("col_co_freq"),t("col_dw"),t("col_conv")])
        self.t_scan.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.t_scan.setAlternatingRowColors(True); self.t_scan.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        ltabs.addTab(self.t_scan, t("tab_scan_res"))
        # 极化激元
        self.t_pol = QTableWidget(0,6)
        self.t_pol.setHorizontalHeaderLabels([t("col_mode"),t("col_branch"),t("col_e_ha"),t("col_e_ha"),t("col_phot"),t("col_mat")])
        self.t_pol.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.t_pol.setAlternatingRowColors(True); self.t_pol.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        ltabs.addTab(self.t_pol,  t("tab_polariton"))
        # 数值汇总
        self.t_summary = QTableWidget(0,4)
        self.t_summary.setHorizontalHeaderLabels([t("col_param"),t("col_value"),t("col_unit"),t("col_note")])
        self.t_summary.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.t_summary.setAlternatingRowColors(True); self.t_summary.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        ltabs.addTab(self.t_summary, t("tab_summary"))

        self.btn_save = QPushButton(t("btn_save_res"))
        self.btn_save.setEnabled(False); self.btn_save.clicked.connect(self._save)
        self.ltabs.setCornerWidget(self.btn_save)
        sp.addWidget(ltabs)

        # 右：图表 Tab
        self.rtabs = QTabWidget()
        rtabs = self.rtabs
        self.lam_plot  = LambdaPlot();  rtabs.addTab(self.lam_plot,  t("tab_lam_plot"))
        self.bar_plot  = FreqBarPlot(); rtabs.addTab(self.bar_plot,  t("tab_shift_plot"))
        sp.addWidget(rtabs)
        sp.setStretchFactor(0,3); sp.setStretchFactor(1,2)
        lay.addWidget(sp,stretch=1)
        self._data: dict = {}

    @staticmethod
    def _card(title, value, color):
        gb = QGroupBox(title)
        gb.setStyleSheet(
            f"QGroupBox{{color:{color};border:1px solid {color}44;border-radius:6px;"
            f"margin-top:8px;padding-top:4px;font-weight:bold;}}"
            f"QGroupBox::title{{subcontrol-origin:margin;left:8px;padding:0 4px;}}")
        ll = QVBoxLayout(gb); lbl = QLabel(value)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setFont(QFont("Monospace",11)); lbl.setStyleSheet(f"color:{color};")
        ll.addWidget(lbl); gb._lbl = lbl; return gb

    def _cv(self, card, txt): card._lbl.setText(txt)

    @Slot(dict)
    def receive(self, data: dict):
        self._data.update(data)
        if e := data.get("total_energy"):
            self._cv(self.c_energy, f"{e:.8f} Ha\n={e*AU2EV:.4f} eV")
        if ae := data.get("ads_energy_ev"):   self._cv(self.c_ads,   f"{ae:+.4f} eV")
        if f0 := data.get("freq_free"):        self._cv(self.c_freq0, f"{f0:.1f} cm⁻¹")
        if df := data.get("freq_shift"):
            self._cv(self.c_dfreq, f"{'+' if df>=0 else ''}{df:.2f} cm⁻¹")
        if (conv := data.get("converged")) is not None:
            self._cv(self.c_conv, t("conv_yes") if conv else t("conv_no"))

        if freqs := data.get("frequencies"):
            self.t_freq.setRowCount(0)
            f0v = data.get("freq_free")
            for i,f in enumerate(freqs,1):
                r = self.t_freq.rowCount(); self.t_freq.insertRow(r)
                is_co = f0v and abs(float(f)-f0v)<5
                for c,v in enumerate([str(i),f"{float(f):.2f}","—","★ C-O" if is_co else ""]):
                    it = QTableWidgetItem(v); it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    if is_co: it.setForeground(QColor("#50fa7b"))
                    self.t_freq.setItem(r,c,it)

        if pols := data.get("polariton_energies"):
            self.t_pol.setRowCount(0)
            for p in pols:
                r=self.t_pol.rowCount(); self.t_pol.insertRow(r)
                for c,v in enumerate([str(p.get("mode","")),str(p.get("branch","")),
                                       f"{p.get('energy_ha',0):.8f}",f"{p.get('energy_ev',0):.4f}",
                                       f"{p.get('photon_weight',0):.4f}",f"{p.get('matter_weight',0):.4f}"]):
                    it = QTableWidgetItem(v); it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.t_pol.setItem(r,c,it)

        self._rebuild_summary(data)
        self.btn_save.setEnabled(True)

    @Slot(dict)
    def append_scan(self, rd: dict):
        lam=rd.get("lambda",0); e=rd.get("total_energy"); ae=rd.get("ads_energy_ev")
        freq=rd.get("co_freq"); dw=rd.get("freq_shift"); conv=rd.get("converged",False)
        r=self.t_scan.rowCount(); self.t_scan.insertRow(r)
        vals=[f"{lam:.4f}", f"{e:.8f}" if e is not None else "—",
              f"{ae:+.4f}" if ae is not None else "—",
              f"{freq:.1f}" if freq is not None else "—",
              f"{dw:+.2f}" if dw is not None else "—",
              "✓" if conv else "✗"]
        for c,v in enumerate(vals):
            it=QTableWidgetItem(v); it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if v=="✗": it.setForeground(QColor("#ff5555"))
            elif v=="✓": it.setForeground(QColor("#50fa7b"))
            self.t_scan.setItem(r,c,it)
        self.t_scan.scrollToBottom()
        self.lam_plot.append_point(lam,ae,freq)

    def _rebuild_summary(self, data):
        self.t_summary.setRowCount(0)
        rows=[(t("sum_total_e"),  data.get("total_energy"),   "Ha",     t("sum_scf_ref")),
              (t("sum_ads_e"),    data.get("ads_energy_ev"),  "eV",     t("sum_ads_ref")),
              (t("sum_freq0"),    data.get("freq_free"),      "cm⁻¹",   t("sum_vib_ref")),
              (t("sum_freq_cav"), data.get("freq_cav"),       "cm⁻¹",   t("sum_cav_ref")),
              (t("sum_shift"),    data.get("freq_shift"),     "cm⁻¹",   t("sum_shift_ref")),
              (t("sum_dipole"),   data.get("dipole_norm"),    "Debye",  ""),
              (t("sum_walltime"), data.get("wall_time"),      "s",      t("sum_pipe_ref"))]
        for param,val,unit,note in rows:
            if val is None: continue
            r=self.t_summary.rowCount(); self.t_summary.insertRow(r)
            fmt=f"{val:.6f}" if isinstance(val,float) else str(val)
            for c,v in enumerate([param,fmt,unit,note]):
                it=QTableWidgetItem(v)
                if c==0: it.setForeground(QColor("#aaaaaa"))
                self.t_summary.setItem(r,c,it)
        # 频移柱状图
        lam_rows=[]
        for r in range(self.t_scan.rowCount()):
            try:
                lt=self.t_scan.item(r,0); dt=self.t_scan.item(r,4)
                if lt and dt and dt.text()!="—": lam_rows.append((float(lt.text()),float(dt.text())))
            except: pass
        if lam_rows:
            self.bar_plot.update_data([f"λ={x[0]:.3f}" for x in lam_rows],[x[1] for x in lam_rows])

    def reset(self):
        for c in (self.c_energy,self.c_ads,self.c_freq0,self.c_dfreq,self.c_conv): c._lbl.setText("—")
        for t in (self.t_freq,self.t_scan,self.t_pol,self.t_summary): t.setRowCount(0)
        self.lam_plot.reset(); self.btn_save.setEnabled(False); self._data={}

    def _save(self):
        if not self._data: return
        p,_=QFileDialog.getSaveFileName(self,t("dlg_save_res"),"nqed_result.json","JSON (*.json)")
        if p:
            with open(p,"w",encoding="utf-8") as f:
                json.dump(self._data,f,indent=2,ensure_ascii=False,
                          default=lambda x: None if (isinstance(x,float) and x!=x) else x)
            QMessageBox.information(self,t("save_ok"),f"{t('save_ok_msg')}\n{p}")

    def retranslate(self):
        self.c_energy._lbl.setParent(None) if False else None  # no-op
        # Cards
        self.c_energy.setTitle(t("card_energy")); self.c_ads.setTitle(t("card_ads"))
        self.c_freq0.setTitle(t("card_freq0")); self.c_dfreq.setTitle(t("card_dfreq"))
        self.c_conv.setTitle(t("card_conv"))
        # Table headers
        self.t_freq.setHorizontalHeaderLabels([t("col_mode"),t("col_freq"),t("col_intens"),t("col_annot")])
        self.t_scan.setHorizontalHeaderLabels([t("col_lam"),t("col_e_ha"),t("col_de_ads"),t("col_co_freq"),t("col_dw"),t("col_conv")])
        self.t_pol.setHorizontalHeaderLabels([t("col_mode"),t("col_branch"),t("col_e_ha"),t("col_e_ha"),t("col_phot"),t("col_mat")])
        self.t_summary.setHorizontalHeaderLabels([t("col_param"),t("col_value"),t("col_unit"),t("col_note")])
        # Sub-tabs
        self.ltabs.setTabText(0,t("tab_vib")); self.ltabs.setTabText(1,t("tab_scan_res"))
        self.ltabs.setTabText(2,t("tab_polariton")); self.ltabs.setTabText(3,t("tab_summary"))
        self.rtabs.setTabText(0,t("tab_lam_plot")); self.rtabs.setTabText(1,t("tab_shift_plot"))
        self.btn_save.setText(t("btn_save_res"))


# ═══════════════════════════════════════════════════════════
#  计算 Worker（后台流水线）
# ═══════════════════════════════════════════════════════════

class CalcWorker(QThread):
    log_line     = Signal(str)
    progress     = Signal(int)
    step_state   = Signal(int,str)
    step_label   = Signal(str)
    scf_step     = Signal(int,float,float)
    scan_point   = Signal(dict)
    result_ready = Signal(dict)
    error        = Signal(str)

    S_GEOM,S_GAS,S_FREQ0,S_SCAN,S_FREQCAV,S_POL = range(6)

    def __init__(self, cfg, atoms, parent=None):
        super().__init__(parent); self._cfg=cfg; self._atoms=atoms
        self._abort=False; self._res={}

    def abort(self): self._abort=True

    def run(self):
        try: self._pipeline()
        except Exception as ex:
            import traceback; self.error.emit(f"{type(ex).__name__}: {ex}\n{traceback.format_exc()}")

    def _l(self,msg): self.log_line.emit(msg)

    def _pipeline(self):
        t0=time.time(); c=self._cfg

        # 断点
        ckpt=c.get("checkpoint","nqed_ckpt.json")
        if c.get("resume") and Path(ckpt).exists():
            with open(ckpt) as f: self._res=json.load(f)
            self._l(f"[RESUME] loaded: {ckpt}")

        use_nqed = c.get("use_nqed",False) and HAS_NQEDDFT
        if use_nqed:
            cluster=c["cluster"]; ads=c["ads"]
            geom_str=build_geom(cluster,ads); spin=calc_spin(cluster,ads)
            basis_d=BASIS_NQED
            self._l(f"[nQEDDFT] {cluster}" + (f" + {ads}" if ads else ""))
        else:
            geom_str="\n".join(f"{s}  {x:.6f}  {y:.6f}  {z:.6f}" for s,x,y,z in self._atoms)
            spin=c["spin"]; basis_d=c.get("basis","6-31G*")
        xc=c["xc"]

        try:
            from pyscf import gto
            mol=gto.M(atom=geom_str,basis=basis_d,spin=spin,charge=c.get("charge",0),
                      unit="Angstrom",verbose=0)
        except Exception as ex: self.error.emit(f"Failed to build molecule: {ex}"); return

        # ① 几何优化
        mol_opt=mol
        if c.get("do_geom_opt") and not self._res.get("_geom_ok"):
            self.step_state.emit(self.S_GEOM,"running"); self.step_label.emit(t("worker_geom"))
            self.progress.emit(5)
            try:
                from pyscf.dft import rks as rks_, uks as uks_
                from pyscf.geomopt import geometric_solver
                mf0=(uks_.UKS if spin else rks_.RKS)(mol)
                mf0.xc=xc; mf0.max_cycle=c.get("max_cycle",300); mf0.conv_tol=c.get("conv_tol",1e-9)
                if spin: mf0.level_shift=c.get("level_shift",0.3); mf0.init_guess="atom"
                mol_opt=geometric_solver.optimize(mf0)
                self._res["_geom_ok"]=True; self._save_ckpt(ckpt)
                self.step_state.emit(self.S_GEOM,"done"); self._l("  ✓ Geometry optimisation done")
            except Exception as ex:
                self._l(f"  ✗ Geometry optimisation failed: {ex}"); self.step_state.emit(self.S_GEOM,"error"); mol_opt=mol
        else: self.step_state.emit(self.S_GEOM,"skip" if not c.get("do_geom_opt") else "done")

        if self._abort: self._finish(t0); return
        self.progress.emit(15)

        # ② 气相参考
        e_ref=self._res.get("_e_ref")
        if c.get("do_gas_ref") and e_ref is None:
            gk=c.get("gas_ref")
            if gk and gk in GAS_REFS:
                self.step_state.emit(self.S_GAS,"running"); self.step_label.emit(t("worker_gas").format(gk=gk))
                self.progress.emit(22)
                try:
                    from pyscf import gto
                    from pyscf.dft import rks as rks_
                    gbasis=BASIS_NQED.get("default","cc-pVDZ") if use_nqed else basis_d
                    mol_g=gto.M(atom=GAS_REFS[gk],basis=gbasis,spin=0,charge=0,unit="Angstrom",verbose=0)
                    if use_nqed:
                        cd=Cavity(); cd.add_mode(0.1,0.0,[0,0,1]); mfg=QEDRKS(mol_g,cd)
                    else: mfg=rks_.RKS(mol_g)
                    mfg.xc=xc; mfg.max_cycle=300
                    e_ref=float(mfg.kernel()); self._res["_e_ref"]=e_ref
                    self._save_ckpt(ckpt); self.step_state.emit(self.S_GAS,"done")
                    self._l(f"  ✓ Gas ref {gk}: {e_ref:.8f} Ha")
                except Exception as ex:
                    self._l(f"  ✗ Gas reference failed: {ex}"); self.step_state.emit(self.S_GAS,"error")
        else: self.step_state.emit(self.S_GAS,"skip" if not c.get("do_gas_ref") else "done")

        if self._abort: self._finish(t0); return
        self.progress.emit(28)

        # ③ 无腔 SCF + 振动
        freq_free=self._res.get("freq_free"); e0=self._res.get("_e0")
        freqs0=self._res.get("_freqs0",[])
        if c.get("do_freq0") and not self._res.get("_freq0_ok"):
            self.step_state.emit(self.S_FREQ0,"running"); self.step_label.emit(t("worker_freq0"))
            self.progress.emit(32)
            try:
                if use_nqed:
                    cd=Cavity(); cd.add_mode(0.1,0.0,[0,0,1])
                    mf0=(QEDUKS if spin else QEDRKS)(mol_opt,cd)
                    if spin: mf0.level_shift=c.get("level_shift",0.2); mf0.init_guess="atom"
                else:
                    from pyscf.dft import rks as rks_, uks as uks_
                    mf0=(uks_.UKS if spin else rks_.RKS)(mol_opt)
                mf0.xc=xc; mf0.max_cycle=c.get("max_cycle",300); mf0.conv_tol=c.get("conv_tol",1e-9)
                e0=float(mf0.kernel()); self._res["_e0"]=e0
                self._res["total_energy"]=e0; self._res["converged"]=bool(getattr(mf0,"converged",True))
                # 裸团簇能量 + 吸附能
                if e_ref is not None and use_nqed and ads:
                    if "_e_slab" not in self._res:
                        from pyscf import gto
                        sg=build_geom(cluster,None); ss=calc_spin(cluster,None)
                        ms=gto.M(atom=sg,basis=basis_d,spin=ss,charge=0,unit="Angstrom",verbose=0)
                        cd2=Cavity(); cd2.add_mode(0.1,0.0,[0,0,1])
                        mfs=(QEDUKS if ss else QEDRKS)(ms,cd2)
                        mfs.xc=xc; mfs.max_cycle=300
                        self._res["_e_slab"]=float(mfs.kernel()); self._save_ckpt(ckpt)
                    ae=(e0-self._res["_e_slab"]-e_ref)*AU2EV
                    self._res["ads_energy_ev"]=ae; self._l(f"  ΔE_ads = {ae:+.4f} eV")
                # 振动
                if use_nqed:
                    from nqeddft.phonon import QEDPhonon
                    ph=QEDPhonon(mf0); hess=ph.numerical_hessian_fast(stepsize=0.005,verbose=False)
                    freqs_raw,_=ph.harmonic_analysis(hess)
                else:
                    from pyscf.prop.freq import rhf as freq_m
                    freqs_raw,_=freq_m.Freq(mf0).kernel()
                freqs0=[float(f) for f in freqs_raw]
                freq_free=max((f for f in freqs0 if 1000<f<3000),default=None)
                self._res.update({"_freqs0":freqs0,"_freq0_ok":True,
                                   "freq_free":freq_free,"frequencies":freqs0})
                self._save_ckpt(ckpt); self.step_state.emit(self.S_FREQ0,"done")
                self._l(f"  ✓ ν₀(C-O) = {freq_free:.1f} cm⁻¹" if freq_free else "  ✓ Vibration done")
                self.result_ready.emit(dict(self._res))
            except Exception as ex:
                self._l(f"  ✗ Free vibration failed: {ex}"); self.step_state.emit(self.S_FREQ0,"error")
        else: self.step_state.emit(self.S_FREQ0,"skip" if not c.get("do_freq0") else "done")

        if self._abort: self._finish(t0); return
        self.progress.emit(48)

        # ④ λ 扫描
        scan_done=self._res.get("_scan_done",{})
        if c.get("do_scan") and use_nqed:
            self.step_state.emit(self.S_SCAN,"running")
            lams=c.get("lambda_list",LAMBDA_PRESETS)
            pol=c.get("polarization",[0.,0.,1.])
            omega_res=cm_to_au(freq_free) if freq_free else c.get("omega_c",0.0856)
            n=len(lams)
            for idx,lam in enumerate(lams):
                if self._abort: break
                lk=f"lam_{lam:.5f}"
                if scan_done.get(lk):
                    pt_cached = dict(scan_done[lk]); pt_cached["lambda"] = lam
                    self.scan_point.emit(pt_cached); continue
                self.progress.emit(48+int(28*idx/max(n,1)))
                self.step_label.emit(t("worker_scan").format(i=idx+1,n=n,lam=lam))
                self._l(f"\n  ── λ={lam:.4f} ({idx+1}/{n}) ──")
                try:
                    from pyscf import gto
                    cv=Cavity(); cv.add_mode(omega_res,lam,pol)
                    ml=gto.M(atom=build_geom(cluster,ads),basis=basis_d,spin=spin,
                              charge=0,unit="Angstrom",verbose=0)
                    mfl=(QEDUKS if spin else QEDRKS)(ml,cv)
                    mfl.xc=xc; mfl.max_cycle=300; mfl.conv_tol=1e-9
                    if spin: mfl.level_shift=0.2; mfl.init_guess="atom"
                    el=float(mfl.kernel()); conv=bool(getattr(mfl,"converged",True))
                    ael=None
                    if e_ref is not None and "_e_slab" in self._res:
                        ael=(el-self._res["_e_slab"]-e_ref)*AU2EV
                    pt={"lambda":lam,"total_energy":el,"ads_energy_ev":ael,"converged":conv}
                    self._l(f"  E={el:.8f} Ha  conv={conv}")
                    scan_done[lk]=pt; self._res["_scan_done"]=scan_done
                    self._save_ckpt(ckpt); self.scan_point.emit(pt)
                except Exception as ex:
                    self._l(f"  ✗ λ={lam:.4f}: {ex}"); self.scan_point.emit({"lambda":lam,"converged":False})
            self.step_state.emit(self.S_SCAN,"done")
        else: self.step_state.emit(self.S_SCAN,"skip")

        if self._abort: self._finish(t0); return
        self.progress.emit(78)

        # ⑤ 腔中振动 → 频移
        if c.get("do_freq_cav") and use_nqed and not self._res.get("_freqcav_ok"):
            self.step_state.emit(self.S_FREQCAV,"running"); self.step_label.emit(t("worker_freqcav"))
            lam_c=c.get("lambda_val",0.02)
            omega_res2=cm_to_au(freq_free) if freq_free else c.get("omega_c",0.0856)
            try:
                from pyscf import gto
                from nqeddft.phonon import QEDPhonon
                cv2=Cavity(); cv2.add_mode(omega_res2,lam_c,c.get("polarization",[0.,0.,1.]))
                mc=gto.M(atom=build_geom(cluster,ads),basis=basis_d,spin=spin,charge=0,unit="Angstrom",verbose=0)
                mfc=(QEDUKS if spin else QEDRKS)(mc,cv2)
                mfc.xc=xc; mfc.max_cycle=300; mfc.conv_tol=1e-9
                if spin: mfc.level_shift=0.2; mfc.init_guess="atom"
                mfc.kernel(); mfc.mol.build(); mfc.kernel()
                ph2=QEDPhonon(mfc); h2=ph2.numerical_hessian_fast(stepsize=0.005,verbose=False)
                fc,_=ph2.harmonic_analysis(h2)
                fc_list=[float(f) for f in fc]
                freq_cav=max((f for f in fc_list if 1000<f<3000),default=None)
                shift=(freq_cav-freq_free) if (freq_cav and freq_free) else None
                self._res.update({"freq_cav":freq_cav,"freq_shift":shift,"_freqcav_ok":True})
                self._save_ckpt(ckpt); self.step_state.emit(self.S_FREQCAV,"done")
                self._l(f"  ✓ ν_cav={freq_cav:.1f} cm⁻¹  Δω={shift:+.2f} cm⁻¹" if freq_cav else "  ✓ Cavity vibration done")
                self.result_ready.emit(dict(self._res))
            except Exception as ex:
                self._l(f"  ✗ Cavity vibration failed: {ex}"); self.step_state.emit(self.S_FREQCAV,"error")
        else: self.step_state.emit(self.S_FREQCAV,"skip")

        # ⑥ 极化激元
        if c.get("do_polariton"):
            self.step_state.emit(self.S_POL,"running"); self.step_label.emit(t("worker_pol"))
            self.progress.emit(93); self._l("  Polariton analysis (requires nqeddft polariton interface)...")
            self.step_state.emit(self.S_POL,"done")
        else: self.step_state.emit(self.S_POL,"skip")

        self._finish(t0)

    def _finish(self, t0):
        self._res["wall_time"]=time.time()-t0
        self.progress.emit(100); self.step_label.emit(t("worker_done"))
        self.result_ready.emit(dict(self._res))

    def _save_ckpt(self, path):
        try:
            tmp=path+".tmp"
            with open(tmp,"w",encoding="utf-8") as f:
                json.dump(self._res,f,indent=2,ensure_ascii=False,
                          default=lambda x: None if (isinstance(x,float) and x!=x) else x)
            Path(tmp).replace(path)
        except Exception as ex: self._l(f"  [CKPT] save failed: {ex}")


# ═══════════════════════════════════════════════════════════
#  外部进程驱动:stage_pipeline.py 的 subprocess Worker
# ═══════════════════════════════════════════════════════════

class StagePipelineWorker(QThread):
    """
    通过 subprocess 调用 stage_pipeline.py,流式读取 stdout 到日志。

    与 CalcWorker 接口尽量对齐(log_line/progress/step_state/error)
    以便 MainWindow._submit 复用同一套连线。

    设计要点:
      - 子进程崩溃不会影响 GUI(独立进程隔离)
      - 中止 = subprocess.terminate(),比 QThread + PySCF 更可靠
      - stdout 行解析: 普通日志直送 log_line;特殊前缀作为信号:
            [PROGRESS] 35.0
            [STEP] 0 stage0 done
            [DONE]
      - stage_pipeline.py 已经按这种格式输出(我们顺便给它加上回调)
    """

    log_line     = Signal(str)
    progress     = Signal(int)
    step_state   = Signal(int, str)
    step_label   = Signal(str)
    task_event   = Signal(dict)
    result_ready = Signal(dict)
    error        = Signal(str)

    # stage 编号 → MonitorPanel 中要点亮的 step idx 列表
    # (复用现有 6 个 step 槽位)
    STAGE_TO_MONITOR = {
        0: [0, 1, 2],   # geom_opt + gas_ref + freq0
        1: [3, 4],      # scan + freq_cav
        2: [5],         # 借用 polariton 槽位
    }

    def __init__(self, stages, workdir, resume=True,
                 stage_script="stage_pipeline.py",
                 python_exe=None, parent=None):
        super().__init__(parent)
        self._stages       = list(stages)
        self._workdir      = str(workdir)
        self._resume       = bool(resume)
        self._stage_script = stage_script
        self._python_exe   = python_exe or sys.executable
        self._proc         = None
        self._abort        = False
        self._status_path  = Path(self._workdir) / "task_status.json"
        self._started_at   = None

    def abort(self):
        """请求中止子进程(由 GUI 的 中止 按钮触发)。"""
        self._abort = True
        if self._proc and self._proc.poll() is None:
            try:
                # 先 terminate (SIGTERM, Windows 上变成 TerminateProcess)
                self._proc.terminate()
            except Exception:
                pass

    def run(self):
        import subprocess, shlex
        try:
            # 解析脚本路径:stage_script 若是相对路径,先按 main_window 同目录找
            script = Path(self._stage_script)
            if not script.is_absolute():
                here = Path(__file__).resolve().parent
                candidates = [
                    Path.cwd() / script,
                    here / script,
                    here.parent / script,
                    Path(self._workdir) / script,
                ]
                script = next((p for p in candidates if p.exists()), script)
            script = script.resolve()
            cmd = [self._python_exe, str(script),
                   "--stages", ",".join(str(s) for s in self._stages),
                   "--workdir", self._workdir]
            if not self._resume:
                cmd.append("--no-resume")

            self.log_line.emit(f"[CMD] {' '.join(shlex.quote(c) for c in cmd)}")
            self.step_label.emit(t("worker_starting"))
            self._started_at = time.time()
            self._write_status("running", script=str(script), command=cmd)

            # 子进程启动
            #   PYTHONUNBUFFERED=1 → 子进程 print 立即 flush,逐行可读
            #   stderr → stdout 合并,简化读取
            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,                # line-buffered
                text=True,
                env=env,
                cwd=self._workdir,
            )

            # 流式读 stdout
            for raw in self._proc.stdout:
                if self._abort:
                    break
                line = raw.rstrip("\r\n")
                self._dispatch_line(line)

            # 等结束 + 拿退出码
            rc = self._proc.wait()
            if self._abort:
                self.log_line.emit(f"[ABORT] subprocess terminated (rc={rc})")
                self.step_label.emit(t("worker_aborted"))
                self._write_status("aborted", rc=rc)
            elif rc != 0:
                self._write_status("error", rc=rc)
                self.error.emit(f"stage_pipeline.py exited with code {rc}")
                return
            else:
                self.log_line.emit("[OK] subprocess exit 0")
                self._write_status("done", rc=rc)

            # 收集断点 json 作为结果(stage0 / stage1 / stage2 共两个文件)
            results = {}
            for fname in ("stage0_results.json", "stage1_2_results.json"):
                fp = Path(self._workdir) / fname
                if fp.exists():
                    try:
                        with open(fp, encoding="utf-8") as f:
                            results[fname] = json.load(f)
                    except Exception as ex:
                        self.log_line.emit(f"  [WARN] 读取 {fname} 失败: {ex}")
            self.progress.emit(100)
            self.result_ready.emit(results)

        except Exception as ex:
            import traceback
            self.error.emit(f"{type(ex).__name__}: {ex}\n"
                            f"{traceback.format_exc()}")

    # ── stdout 行解析 ─────────────────────────────────────────────
    def _dispatch_line(self, line: str) -> None:
        """识别特殊前缀,其余作为日志透传。"""
        # [PROGRESS] 42.5
        if line.startswith("[PROGRESS]"):
            try:
                pct = float(line.split(None, 1)[1])
                self.progress.emit(int(pct))
            except Exception:
                self.log_line.emit(line)
            return
        # [STEP] 0 stage0 start|done|skip|error
        if line.startswith("[STEP]"):
            try:
                parts = line.split(None, 3)
                idx = int(parts[1])
                state = parts[3] if len(parts) > 3 else "running"
                # 把 stage idx 映射成 monitor 的 step idx
                for mon_idx in self.STAGE_TO_MONITOR.get(idx, []):
                    self.step_state.emit(
                        mon_idx,
                        "running" if state == "start" else state,
                    )
                # stage 名也送一下,刷新顶部状态
                if len(parts) > 2:
                    self.step_label.emit(f"{parts[2]} — {state}")
                    self.task_event.emit({
                        "idx": idx,
                        "name": parts[2],
                        "state": "running" if state == "start" else state,
                        "workdir": self._workdir,
                    })
                    self._write_status("running", last_stage=idx,
                                       last_stage_name=parts[2],
                                       last_stage_state=state)
            except Exception:
                self.log_line.emit(line)
            return
        # 普通日志
        self.log_line.emit(line)

    def _write_status(self, state: str, **extra) -> None:
        try:
            p = Path(self._workdir)
            p.mkdir(parents=True, exist_ok=True)
            data = {
                "state": state,
                "stages": self._stages,
                "workdir": str(p.resolve()),
                "resume": self._resume,
                "started_at": self._started_at,
                "updated_at": time.time(),
                "elapsed_s": (time.time() - self._started_at) if self._started_at else None,
            }
            data.update(extra)
            tmp = self._status_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._status_path)
        except Exception as ex:
            self.log_line.emit(f"  [WARN] failed to write task_status.json: {ex}")


# ═══════════════════════════════════════════════════════════
#  提交命令对话框
# ═══════════════════════════════════════════════════════════

CMD_TMPLS = {
    "local": "python -m qed_dft_studio.run --method {method} --basis {basis} --input {input_file} --output {output_file}",
    "PBS":   "qsub -l nodes=1:ppn={ncpus},mem={mem}gb,walltime={walltime} -v METHOD={method},INPUT={input_file} run.pbs",
    "SLURM": "sbatch --ntasks={ncpus} --mem={mem}G --time={walltime} --export=METHOD={method},INPUT={input_file} run.slurm",
    "custom": "",
}

class CommandEditorDialog(QDialog):
    def __init__(self, tmpl="", parent=None):
        super().__init__(parent); self.setWindowTitle(t("cmd_dlg_title")); self.resize(680,420)
        ll=QVBoxLayout(self)
        self.lbl_tmpl=QLabel(t("cmd_lbl_template")); row=QHBoxLayout(); row.addWidget(self.lbl_tmpl)
        self.combo=QComboBox()
        self.combo.addItem(t("cmd_local"), "local")
        self.combo.addItem("PBS", "PBS")
        self.combo.addItem("SLURM", "SLURM")
        self.combo.addItem(t("cmd_custom"), "custom")
        row.addWidget(self.combo,1); ll.addLayout(row)
        self.var_t=QTableWidget(0,2)
        self.var_t.setHorizontalHeaderLabels([t("cmd_col_var"),t("cmd_col_val")])
        self.var_t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.var_t.setAlternatingRowColors(True); ll.addWidget(self.var_t)
        self.cmd_e=QPlainTextEdit(); self.cmd_e.setFont(QFont("Courier New",9))
        self.cmd_e.setMinimumHeight(80); self.cmd_e.setPlainText(tmpl); ll.addWidget(self.cmd_e)
        ll.addWidget(QLabel("{method} {basis} {charge} {spin} {ncpus} {mem} {walltime} {input_file} {output_file} {jobname}",
                            styleSheet="color:#888888;font-size:8pt;"))
        bb=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self.accept); bb.rejected.connect(self.reject); ll.addWidget(bb)
        self._v={"method":"DFT","basis":"6-31G*","charge":"0","spin":"0","ncpus":"8",
                 "mem":"32","walltime":"24:00:00","input_file":"mol.xyz","output_file":"res.json","jobname":"calc001"}
        self._pop(); self.combo.currentTextChanged.connect(lambda n: self.cmd_e.setPlainText(CMD_TMPLS.get(n,"")))

    def _pop(self):
        self.var_t.setRowCount(0)
        for k,v in self._v.items():
            r=self.var_t.rowCount(); self.var_t.insertRow(r)
            ki=QTableWidgetItem(k); ki.setFlags(ki.flags()&~Qt.ItemFlag.ItemIsEditable); ki.setForeground(QColor("#aaaaaa"))
            self.var_t.setItem(r,0,ki); self.var_t.setItem(r,1,QTableWidgetItem(v))

    def get_cmd(self):
        cmd=self.cmd_e.toPlainText()
        for r in range(self.var_t.rowCount()):
            k=self.var_t.item(r,0); v=self.var_t.item(r,1)
            if k and v: cmd=cmd.replace("{"+k.text()+"}",v.text())
        return cmd
    def get_tmpl(self): return self.cmd_e.toPlainText()


class WorkflowBar(QWidget):
    """Compact top-level workflow indicator for the four main tabs."""

    _KEYS = ("flow_struct", "flow_settings", "flow_monitor", "flow_results")

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 4)
        lay.setSpacing(8)
        self._labels: list[QLabel] = []
        self._seps: list[QLabel] = []
        for i, key in enumerate(self._KEYS):
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setMinimumHeight(30)
            lay.addWidget(lbl, 1)
            self._labels.append(lbl)
            if i < len(self._KEYS) - 1:
                sep = QLabel("›")
                sep.setAlignment(Qt.AlignmentFlag.AlignCenter)
                sep.setFixedWidth(14)
                sep.setStyleSheet("color:#5f6b7a;font-size:14pt;")
                lay.addWidget(sep)
                self._seps.append(sep)
        self._current = 0
        self._done: set[int] = set()
        self.retranslate()

    def set_current(self, idx: int):
        self._current = max(0, min(idx, len(self._labels) - 1))
        self._refresh()

    def set_done(self, idx: int, done: bool = True):
        if done:
            self._done.add(idx)
        else:
            self._done.discard(idx)
        self._refresh()

    def reset(self):
        self._done.clear()
        self._current = 0
        self._refresh()

    def retranslate(self):
        for i, (lbl, key) in enumerate(zip(self._labels, self._KEYS), 1):
            lbl.setText(f"{i}. {t(key)}")
        self._refresh()

    def _refresh(self):
        for i, lbl in enumerate(self._labels):
            state = "current" if i == self._current else "done" if i in self._done else "idle"
            lbl.setProperty("flowState", state)
            lbl.style().unpolish(lbl)
            lbl.style().polish(lbl)


# ═══════════════════════════════════════════════════════════
#  主窗口
# ═══════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QED-DFT Studio  v1.3")
        self.resize(1280,860); self._worker=None
        self._cmd_tmpl=CMD_TMPLS["local"]
        self._project_path = None        # 当前项目 .json 路径(用于保存覆盖)
        self._build_menu(); self._build_toolbar()
        self._build_central(); self._build_statusbar()
        self._apply_style()
        self._load_icon()
        nqed=t("status_nqed_ok") if HAS_NQEDDFT else t("status_nqed_miss")
        self.status_lbl.setText(f"{t('status_ready')}  ·  {nqed}")
        self._update_title()

    def _build_menu(self):
        mb=self.menuBar()
        # 文件
        self.menu_file=mb.addMenu(t("menu_file"))
        self.act_new    = QAction(t("menu_new"),   self); self.act_new.triggered.connect(self._new_proj);  self.menu_file.addAction(self.act_new)
        self.act_open   = QAction(t("menu_open"),  self); self.act_open.triggered.connect(self._open_proj); self.menu_file.addAction(self.act_open)
        self.act_save   = QAction(t("menu_save"),  self); self.act_save.triggered.connect(self._save_proj); self.menu_file.addAction(self.act_save)
        self.act_export = QAction(t("menu_export"),self); self.act_export.triggered.connect(self._export);  self.menu_file.addAction(self.act_export)
        self.menu_file.addSeparator()
        self.act_exit   = QAction(t("menu_exit"),  self); self.act_exit.triggered.connect(self.close); self.menu_file.addAction(self.act_exit)
        # 计算
        self.menu_calc=mb.addMenu(t("menu_calc"))
        self.act_sub=QAction(t("menu_submit"),self); self.act_sub.triggered.connect(self._submit)
        self.act_abt=QAction(t("menu_abort"), self); self.act_abt.triggered.connect(self._abort); self.act_abt.setEnabled(False)
        self.act_cmd=QAction(t("menu_cmd"),   self); self.act_cmd.triggered.connect(self._edit_cmd)
        self.menu_calc.addAction(self.act_sub); self.menu_calc.addAction(self.act_abt)
        self.menu_calc.addSeparator(); self.menu_calc.addAction(self.act_cmd)
        # 视图
        self.menu_view=mb.addMenu(t("menu_view"))
        self.menu_lang=self.menu_view.addMenu("Language / 语言")
        self.act_lang_zh=QAction(t("menu_lang_zh"),self,checkable=True)
        self.act_lang_en=QAction(t("menu_lang_en"),self,checkable=True)
        self.act_lang_zh.setChecked(LANG=="zh"); self.act_lang_en.setChecked(LANG=="en")
        self.act_lang_zh.triggered.connect(lambda: self._set_lang("zh"))
        self.act_lang_en.triggered.connect(lambda: self._set_lang("en"))
        self.menu_lang.addAction(self.act_lang_zh); self.menu_lang.addAction(self.act_lang_en)
        # 帮助
        self.menu_help=mb.addMenu(t("menu_help"))
        self.act_about=QAction(t("menu_about"),self); self.act_about.triggered.connect(self._about)
        self.menu_help.addAction(self.act_about)

    def _build_toolbar(self):
        self._tb=self.addToolBar("main"); self._tb.setMovable(False)
        self.tb_sub=QPushButton(t("tb_submit")); self.tb_sub.setObjectName("btn_submit"); self.tb_sub.clicked.connect(self._submit)
        self.tb_abt=QPushButton(t("tb_abort"));  self.tb_abt.setObjectName("btn_abort");  self.tb_abt.setEnabled(False); self.tb_abt.clicked.connect(self._abort)
        self.tb_sav=QPushButton(t("tb_save"));   self.tb_sav.clicked.connect(self._save_proj)
        self.tb_cmd=QPushButton(t("tb_cmd"));    self.tb_cmd.clicked.connect(self._edit_cmd)
        # 语言快速切换按钮（右侧）
        from PySide6.QtWidgets import QToolButton
        self.tb_lang=QToolButton(); self.tb_lang.setText("EN" if LANG=="zh" else "中")
        self.tb_lang.setToolTip("Switch Language / 切换语言")
        self.tb_lang.setFixedWidth(36); self.tb_lang.clicked.connect(self._toggle_lang)
        self.tb_lang.setStyleSheet("QToolButton{background:#1e2a3a;color:#4d9fff;border:1px solid #2a4a6a;"
                                   "border-radius:4px;font-weight:bold;padding:4px;}")
        for w in (self.tb_sub,self.tb_abt,self.tb_sav,self.tb_cmd): self._tb.addWidget(w)
        self._tb.addSeparator()
        self._tb.addWidget(self.tb_lang)

    def _build_central(self):
        central = QWidget()
        central_lay = QVBoxLayout(central)
        central_lay.setContentsMargins(6, 4, 6, 6)
        central_lay.setSpacing(4)
        self.flow = WorkflowBar()
        central_lay.addWidget(self.flow)
        self.tabs=QTabWidget()
        central_lay.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.struct=StructurePanel(); self.tabs.addTab(self.struct,t("tab_struct"))
        self.settings=CalcSettingsPanel()
        self.settings.load_cluster.connect(self._on_load_cluster)
        self.tabs.addTab(self.settings,t("tab_settings"))
        self.monitor=MonitorPanel(); self.tabs.addTab(self.monitor,t("tab_monitor"))
        self.results=ResultsPanel(); self.tabs.addTab(self.results,t("tab_results"))
        self.tabs.currentChanged.connect(self.flow.set_current)
        self.struct.atoms_changed.connect(lambda atoms: self.flow.set_done(0, bool(atoms)))

    def _build_statusbar(self):
        sb=QStatusBar(); self.setStatusBar(sb)
        self.status_lbl=QLabel(t("status_ready"))
        self.sb_prog=QProgressBar(); self.sb_prog.setRange(0,100); self.sb_prog.setFixedWidth(160); self.sb_prog.setVisible(False)
        sb.addWidget(self.status_lbl); sb.addPermanentWidget(self.sb_prog)

    @Slot(list,str)
    def _on_load_cluster(self,atoms,label):
        self.struct.set_atoms(atoms,label); self.tabs.setCurrentIndex(0)

    def _stage_names(self, stages: list[int]) -> list[str]:
        labels = {
            0: "stage0 — references / optimisation / vibration",
            1: "stage1 — λ reference and scan",
            2: "stage2 — ω detuning scan",
        }
        return [labels.get(s, f"stage{s}") for s in stages]

    def _resolve_stage_script(self, script_text: str) -> Path | None:
        script = Path(script_text or "stage_pipeline.py")
        if script.is_absolute() and script.exists():
            return script
        here = Path(__file__).resolve().parent
        candidates = [
            Path.cwd() / script,
            here / script,
            here.parent / script,
            Path(script_text),
        ]
        return next((p.resolve() for p in candidates if p.exists()), None)

    def _preflight(self, cfg: dict, atoms: list) -> tuple[bool, list[str], Path | None, Path | None]:
        errors: list[str] = []
        workdir = Path(cfg.get("stage_workdir") or "./run").expanduser()
        script_path = None
        if cfg.get("use_stages"):
            if not cfg.get("stages"):
                errors.append(t("warn_no_stages"))
            script_path = self._resolve_stage_script(cfg.get("stage_script") or "stage_pipeline.py")
            if script_path is None:
                errors.append(f"{t('warn_stage_script')} {cfg.get('stage_script') or 'stage_pipeline.py'}")
            try:
                workdir.mkdir(parents=True, exist_ok=True)
                probe = workdir / ".write_test"
                probe.write_text("ok", encoding="utf-8")
                probe.unlink(missing_ok=True)
            except Exception as ex:
                errors.append(f"{t('warn_bad_workdir')} {workdir} ({ex})")
        elif not cfg.get("use_nqed") and not atoms:
            errors.append(t("warn_no_atoms"))
        if cfg.get("do_scan"):
            try:
                lams = cfg.get("lambda_list") or []
                if not lams or any(float(x) < 0 for x in lams):
                    errors.append(t("warn_bad_lambda"))
            except Exception:
                errors.append(t("warn_bad_lambda"))
        return not errors, errors, workdir.resolve(), script_path

    def _write_task_manifest(self, workdir: Path, cfg: dict, atoms: list, script_path: Path | None = None) -> None:
        manifest = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "app": "QED-DFT Studio",
            "config": cfg,
            "atoms_count": len(atoms),
            "script": str(script_path) if script_path else None,
            "input_xyz": "input.xyz" if atoms else None,
        }
        p = workdir / "task_manifest.json"
        p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        self.monitor.append_log(f"[TASK] {t('task_manifest_saved')} {p}")

    @Slot()
    def _submit(self):
        atoms = self.struct.get_atoms()
        cfg   = self.settings.get_config()
        ok, errors, workdir_path, script_path = self._preflight(cfg, atoms)
        if not ok:
            QMessageBox.warning(self, t("warn_title"),
                                t("warn_preflight") + "\n\n" + "\n".join(f"• {e}" for e in errors))
            return

        # ── 模式 A: stage_pipeline.py 子进程 ──────────────────────
        if cfg.get("use_stages"):
            stages = cfg.get("stages") or []
            workdir = str(workdir_path)
            script  = str(script_path)
            resume  = cfg.get("resume", True)

            # 如果有 atoms,先把它们写到 workdir 的 input.xyz 让 stage 用
            try:
                Path(workdir).mkdir(parents=True, exist_ok=True)
                if atoms:
                    xyz_p = Path(workdir) / "input.xyz"
                    lines = [str(len(atoms)),
                             f"Submitted by QED-DFT Studio"]
                    for sym, x, y, z in atoms:
                        lines.append(
                            f"{sym:<4s}  {x:12.6f}  {y:12.6f}  {z:12.6f}")
                    xyz_p.write_text("\n".join(lines) + "\n",
                                      encoding="utf-8")
            except Exception as ex:
                logger.warning(f"无法写 input.xyz: {ex}")

            self.monitor.reset()
            self.results.reset()
            self.settings.scan_prev.reset()
            self.monitor.seed_tasks(self._stage_names(stages), workdir)
            self.tabs.setCurrentIndex(2)
            self.flow.set_done(1, True)
            self._write_task_manifest(Path(workdir), cfg, atoms, script_path)

            self._worker = StagePipelineWorker(
                stages       = stages,
                workdir      = workdir,
                resume       = resume,
                stage_script = script,
            )
            self._worker.log_line.connect(self.monitor.append_log)
            self._worker.progress.connect(self.monitor.set_prog)
            self._worker.progress.connect(self.sb_prog.setValue)
            self._worker.step_state.connect(self.monitor.set_step)
            self._worker.step_label.connect(self.monitor.set_label)
            self._worker.step_label.connect(self.status_lbl.setText)
            self._worker.task_event.connect(self.monitor.update_task_event)
            self._worker.result_ready.connect(self.results.receive)
            self._worker.result_ready.connect(lambda _: self.flow.set_done(2, True))
            self._worker.result_ready.connect(lambda _: self.flow.set_done(3, True))
            self._worker.result_ready.connect(
                lambda _: self.tabs.setCurrentIndex(3))
            self._worker.finished.connect(self._done)
            self._worker.error.connect(self._err)
            self._worker.start()
            self.tb_sub.setEnabled(False); self.tb_abt.setEnabled(True)
            self.act_sub.setEnabled(False); self.act_abt.setEnabled(True)
            self.sb_prog.setVisible(True)
            return

        # ── 模式 B: 原 CalcWorker(细粒度内置 SCF 流程)─────────
        self.monitor.reset(); self.results.reset(); self.settings.scan_prev.reset()
        self.monitor.seed_tasks([
            t("step_geom"), t("step_gas"), t("step_freq0"),
            t("step_scan"), t("step_freqcav"), t("step_pol")
        ], str(Path(cfg.get("checkpoint", "nqed_ckpt.json")).parent))
        self.tabs.setCurrentIndex(2)
        self.flow.set_done(1, True)
        self._worker=CalcWorker(cfg,atoms)
        self._worker.log_line.connect(self.monitor.append_log)
        self._worker.progress.connect(self.monitor.set_prog)
        self._worker.progress.connect(self.sb_prog.setValue)
        self._worker.step_state.connect(self.monitor.set_step)
        self._worker.step_state.connect(
            lambda idx, state: self.monitor.update_task(
                idx,
                t(MonitorPanel._STEP_KEYS[idx]) if 0 <= idx < len(MonitorPanel._STEP_KEYS) else f"step{idx}",
                state,
                str(Path(cfg.get("checkpoint", "nqed_ckpt.json")).parent),
            )
        )
        self._worker.step_label.connect(self.monitor.set_label)
        self._worker.step_label.connect(self.status_lbl.setText)
        self._worker.scf_step.connect(self.monitor.scf_plot.add_point)
        self._worker.scan_point.connect(self.results.append_scan)
        self._worker.scan_point.connect(self._fwd_scan)
        self._worker.result_ready.connect(self.results.receive)
        self._worker.result_ready.connect(lambda _: self.flow.set_done(2, True))
        self._worker.result_ready.connect(lambda _: self.flow.set_done(3, True))
        self._worker.result_ready.connect(lambda _: self.tabs.setCurrentIndex(3))
        self._worker.finished.connect(self._done)
        self._worker.error.connect(self._err)
        self._worker.start()
        self.tb_sub.setEnabled(False); self.tb_abt.setEnabled(True)
        self.act_sub.setEnabled(False); self.act_abt.setEnabled(True)
        self.sb_prog.setVisible(True)

    @Slot(dict)
    def _fwd_scan(self,pt):
        self.settings.scan_prev.append_point(pt.get("lambda",0),pt.get("ads_energy_ev"),pt.get("co_freq"))

    @Slot()
    def _abort(self):
        if self._worker and self._worker.isRunning():
            self._worker.abort(); self._worker.wait(3000)
        self._done()

    @Slot()
    def _done(self,*_):
        self.tb_sub.setEnabled(True); self.tb_abt.setEnabled(False)
        self.act_sub.setEnabled(True); self.act_abt.setEnabled(False)
        self.sb_prog.setVisible(False); self.status_lbl.setText(t("status_ready"))

    @Slot(str)
    def _err(self,msg): self._done(); QMessageBox.critical(self,t("err_title"),msg)

    @Slot()
    def _edit_cmd(self):
        dlg=CommandEditorDialog(self._cmd_tmpl,self)
        if dlg.exec()==QDialog.DialogCode.Accepted:
            self._cmd_tmpl=dlg.get_tmpl(); self.status_lbl.setText(f"{t('cmd_updated')} {dlg.get_cmd()[:50]}...")

    @Slot()
    def _new_proj(self):
        """新建项目: 清空结构 / 结果 / 监控,设置回默认。"""
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, t("warn_title"), t("warn_busy"))
            return
        # 软确认: 当前有原子或有结果时才提示
        has_data = bool(self.struct.get_atoms()) or bool(self.results._data)
        if has_data:
            r = QMessageBox.question(
                self, t("dlg_new_title"), t("dlg_new_confirm"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if r != QMessageBox.StandardButton.Yes:
                return
        # 清空结构 / 结果 / 监控
        self.struct.set_atoms([], t("struct_empty_hint"))
        self.results.reset()
        self.monitor.reset()
        self.settings.scan_prev.reset()
        self._project_path = None
        self.flow.reset()
        self._update_title()
        self.tabs.setCurrentIndex(0)
        self.status_lbl.setText(t("proj_new_done"))

    @Slot()
    def _open_proj(self):
        """打开项目: 从 JSON 恢复 atoms + config + results。

        atoms 数据流(单一权威源 = .xyz 文件):
          - 新版项目(v1.4+):atoms_xyz 字段指向同目录 .xyz 文件,
            atoms 完全来自该 .xyz;.xyz 丢失 → 警告 + 加载空结构。
          - 老版项目(v1.3):atoms_xyz 字段不存在,atoms 来自 json
            里的 atoms 列表(向后兼容,只保留这一种 fallback)。
        """
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, t("warn_title"), t("warn_busy"))
            return
        p, _ = QFileDialog.getOpenFileName(
            self, t("dlg_open_proj"), "", "JSON (*.json);;All (*)")
        if not p:
            return
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as ex:
            QMessageBox.critical(self, t("err_title"),
                                 f"{t('err_proj_load')}\n{ex}")
            return

        proj_path = Path(p)
        atoms: list = []
        xyz_used = False
        is_legacy = "atoms_xyz" not in data    # v1.3 老项目无此字段

        if is_legacy:
            # 老版 v1.3 项目:从 json.atoms 字段直接读
            snap = data.get("atoms", [])
            if isinstance(snap, list):
                atoms = [tuple(a) for a in snap]
        else:
            # 新版 v1.4+:atoms 严格来自 .xyz,缺则空 + 警告
            xyz_rel = data.get("atoms_xyz") or ""
            if xyz_rel:
                xyz_path = (proj_path.parent / xyz_rel
                            if not Path(xyz_rel).is_absolute()
                            else Path(xyz_rel))
                if xyz_path.exists():
                    try:
                        text = xyz_path.read_text(encoding="utf-8",
                                                   errors="replace")
                        atoms = StructurePanel._parse_xyz_text(text)
                        xyz_used = True
                    except Exception as ex:
                        QMessageBox.warning(
                            self, t("warn_title"),
                            f"{t('err_xyz_external')} {xyz_path}\n{ex}\n\n"
                            f"{t('warn_xyz_load_empty')}")
                        atoms = []
                else:
                    # 项目声明了 .xyz 但文件不在 → 警告 + 空
                    QMessageBox.warning(
                        self, t("warn_title"),
                        f"{t('warn_xyz_missing')} {xyz_path}\n\n"
                        f"{t('warn_xyz_load_empty')}")
                    atoms = []
            # else: atoms_xyz 是空字符串(空结构项目),atoms 保持 []

        # 加载结构
        if xyz_used:
            label = f"Project: {proj_path.name}  (← {xyz_rel})"
        elif is_legacy:
            label = f"Project (v1.3): {proj_path.name}"
        elif atoms:
            label = f"Project: {proj_path.name}"
        else:
            label = f"Project: {proj_path.name}  ({t('struct_empty_hint')})"
        self.struct.set_atoms(atoms, label)

        # config
        cfg = data.get("config", {})
        if cfg:
            try:
                self.settings.set_config(cfg)
            except Exception as ex:
                logger.warning(f"set_config: {ex}")

        # results
        results = data.get("results")
        if results:
            try:
                self.results.reset()
                self.results.receive(results)
                self.tabs.setCurrentIndex(3)
            except Exception as ex:
                logger.warning(f"results restore: {ex}")
        else:
            self.results.reset()
            self.tabs.setCurrentIndex(0)

        self.monitor.reset()
        self.flow.set_done(0, bool(atoms))
        self.flow.set_done(1, bool(cfg))
        self.flow.set_done(2, False)
        self.flow.set_done(3, bool(results))
        self._project_path = proj_path
        self._update_title()
        if xyz_used:
            self.status_lbl.setText(
                f"{t('proj_loaded_xyz')} {Path(xyz_rel).name}")
        else:
            self.status_lbl.setText(f"{t('proj_loaded')} {proj_path.name}")

    @Slot()
    def _export(self):
        """导出当前结果到 JSON(纯结果,不含 config / atoms)。"""
        if not self.results._data:
            QMessageBox.information(self, t("warn_title"), t("warn_no_results"))
            return
        p, _ = QFileDialog.getSaveFileName(
            self, t("dlg_export"), "results.json",
            "JSON (*.json);;All (*)")
        if not p:
            return
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self.results._data, f, indent=2,
                          ensure_ascii=False,
                          default=lambda x: None
                                if (isinstance(x, float) and x != x) else x)
        except Exception as ex:
            QMessageBox.critical(self, t("err_title"),
                                 f"{t('err_export')}\n{ex}")
            return
        self.status_lbl.setText(f"{t('export_done')} {Path(p).name}")

    def _update_title(self):
        if self._project_path:
            self.setWindowTitle(f"{t('app_title')} — {self._project_path.name}")
        else:
            self.setWindowTitle(f"{t('app_title')} — [Untitled]")

    @Slot()
    def _save_proj(self):
        # 已有项目路径 → 直接覆盖;否则弹另存为
        if getattr(self, "_project_path", None):
            p = str(self._project_path)
        else:
            p, _ = QFileDialog.getSaveFileName(
                self, t("dlg_save_proj"), "project.json", "JSON (*.json)")
            if not p:
                return

        proj_path = Path(p)
        atoms = self.struct.get_atoms()

        # ── 把 atoms 写到同名 .xyz 文件 ────────────────────────────
        # 设计:atoms 数据 *只* 存在 .xyz 里,json 仅记录 .xyz 文件名;
        #       这样外部编辑 .xyz(VMD / Avogadro / 记事本)在下次
        #       打开项目时会自动生效。.xyz 丢失则提示用户找回。
        xyz_rel = ""
        if atoms:
            xyz_path = proj_path.with_suffix(".xyz")
            try:
                lines = [str(len(atoms)),
                         f"Saved by QED-DFT Studio — {proj_path.stem}"]
                for sym, x, y, z in atoms:
                    lines.append(f"{sym:<4s}  {x:12.6f}  {y:12.6f}  {z:12.6f}")
                xyz_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                xyz_rel = xyz_path.name        # 相对路径 = 文件名(同目录)
            except Exception as ex:
                QMessageBox.warning(self, t("warn_title"),
                                    f"{t('warn_xyz_write_fail')} {ex}")
                return

        # ── 写 .json(不再保存 atoms 快照,避免与 .xyz 不同步)──
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"version":   "1.4",
                           "atoms_xyz": xyz_rel,
                           "config":    self.settings.get_config(),
                           "results":   self.results._data},
                          f, indent=2, ensure_ascii=False,
                          default=lambda x: None
                                if (isinstance(x, float) and x != x) else x)
        except Exception as ex:
            QMessageBox.critical(self, t("err_title"),
                                 f"{t('err_proj_save')}\n{ex}")
            return

        self._project_path = proj_path
        self._update_title()
        msg = t("proj_saved") + f" {proj_path.name}"
        if xyz_rel:
            msg += f"  (+ {xyz_rel})"
        self.status_lbl.setText(msg)

    @Slot()
    def _about(self):
        nqed = t("about_nqed_ok") if HAS_NQEDDFT else t("about_nqed_miss")
        body = t("about_body")
        nqed_label = "<b>nQEDDFT:</b>"
        QMessageBox.about(self, t("about_title"), body + f"{nqed_label}{nqed}")


    @Slot()
    def _toggle_lang(self):
        """工具栏快速切换按钮。"""
        self._set_lang("en" if LANG=="zh" else "zh")

    def _set_lang(self, lang: str):
        global LANG
        LANG = lang
        self.act_lang_zh.setChecked(lang=="zh"); self.act_lang_en.setChecked(lang=="en")
        self.tb_lang.setText("EN" if lang=="zh" else "中")
        self._retranslate()

    def _retranslate(self):
        """刷新整个窗口所有 UI 文本。"""
        self.setWindowTitle(t("app_title"))
        # Tab 标签
        self.tabs.setTabText(0, t("tab_struct"))
        self.tabs.setTabText(1, t("tab_settings"))
        self.tabs.setTabText(2, t("tab_monitor"))
        self.tabs.setTabText(3, t("tab_results"))
        # 菜单
        self.menu_file.setTitle(t("menu_file")); self.menu_calc.setTitle(t("menu_calc"))
        self.menu_view.setTitle(t("menu_view")); self.menu_help.setTitle(t("menu_help"))
        self.act_new.setText(t("menu_new")); self.act_open.setText(t("menu_open"))
        self.act_save.setText(t("menu_save")); self.act_export.setText(t("menu_export"))
        self.act_exit.setText(t("menu_exit")); self.act_sub.setText(t("menu_submit"))
        self.act_abt.setText(t("menu_abort")); self.act_cmd.setText(t("menu_cmd"))
        self.act_about.setText(t("menu_about"))
        self.act_lang_zh.setText(t("menu_lang_zh")); self.act_lang_en.setText(t("menu_lang_en"))
        # 工具栏
        self.tb_sub.setText(t("tb_submit")); self.tb_abt.setText(t("tb_abort"))
        self.tb_sav.setText(t("tb_save"));   self.tb_cmd.setText(t("tb_cmd"))
        # 子面板
        self.struct.retranslate()
        self.settings.retranslate()
        self.monitor.retranslate()
        self.results.retranslate()
        self.flow.retranslate()
        # 状态栏
        nqed=t("status_nqed_ok") if HAS_NQEDDFT else t("status_nqed_miss")
        self.status_lbl.setText(f"{t('status_ready')}  ·  {nqed}")

    def _load_icon(self):
        """从同目录下的 icon.png 加载窗口图标；若文件不存在则跳过。"""
        icon_path = Path(__file__).parent / "icon.png"
        if icon_path.exists():
            from PySide6.QtGui import QIcon
            icon = QIcon(str(icon_path))
            self.setWindowIcon(icon)
            QApplication.instance().setWindowIcon(icon)
        else:
            logger.debug(f"icon.png not found at {icon_path}, skipping.")

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow,QDialog{background:#111827;}
            QWidget{background:#111827;color:#f3f4f6;}

            QLabel[flowState="idle"]{background:#18212f;color:#8fa0b3;border:1px solid #263244;
                                      border-radius:5px;padding:5px 10px;font-weight:bold;}
            QLabel[flowState="current"]{background:#1f3b57;color:#ffffff;border:1px solid #3b82f6;
                                         border-radius:5px;padding:5px 10px;font-weight:bold;}
            QLabel[flowState="done"]{background:#17362d;color:#8ee6bc;border:1px solid #2f8f6a;
                                     border-radius:5px;padding:5px 10px;font-weight:bold;}

            QTabWidget::pane{border:1px solid #253244;background:#141c2a;border-radius:5px;}
            QTabBar::tab{background:#18212f;color:#cbd5e1;padding:6px 18px;margin-right:2px;
                         border-top-left-radius:4px;border-top-right-radius:4px;
                         border:1px solid #253244;border-bottom:none;}
            QTabBar::tab:selected{background:#1f2937;color:#ffffff;font-weight:bold;
                                   border-top:2px solid #3b82f6;}
            QTabBar::tab:hover:!selected{background:#202b3b;}

            QGroupBox{color:#f3f4f6;border:1px solid #253244;border-radius:5px;
                      margin-top:10px;padding-top:6px;font-weight:bold;}
            QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 5px;
                              color:#f3f4f6;background:#111827;}
            QGroupBox::indicator{width:14px;height:14px;}

            QLabel{color:#ffffff;background:transparent;}

            QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox,QTextEdit{
                background:#172033;color:#f3f4f6;
                border:1px solid #2a3a50;border-radius:4px;padding:3px 7px;}
            QLineEdit:focus,QComboBox:focus,QSpinBox:focus,QDoubleSpinBox:focus{
                border:1px solid #3b82f6;}
            QLineEdit:disabled,QComboBox:disabled,QSpinBox:disabled,QDoubleSpinBox:disabled{
                background:#111827;color:#5f6b7a;border-color:#1f2937;}
            QComboBox::drop-down{border:none;width:20px;}
            QComboBox QAbstractItemView{background:#141414;color:#ffffff;
                                         border:1px solid #2a2a2a;selection-background-color:#1e3a5a;}
            QSpinBox::up-button,QSpinBox::down-button,
            QDoubleSpinBox::up-button,QDoubleSpinBox::down-button{
                background:#1e1e1e;border:none;width:16px;}

            QTableWidget{background:#111827;alternate-background-color:#151f2e;
                          color:#f3f4f6;gridline-color:#253244;border:1px solid #253244;}
            QTableWidget::item:selected{background:#24476b;color:#ffffff;}
            QHeaderView::section{background:#18212f;color:#aebacc;
                                   padding:5px 4px;border:none;border-right:1px solid #253244;
                                   font-weight:bold;}

            QPlainTextEdit{background:#0f1724;color:#d8e6f3;
                           font-family:"Courier New",monospace;font-size:9pt;
                           border:1px solid #253244;border-radius:4px;}

            QScrollArea{border:none;background:transparent;}
            QScrollBar:vertical{background:#0a0a0a;width:10px;border:none;}
            QScrollBar::handle:vertical{background:#2a2a2a;border-radius:5px;min-height:20px;}
            QScrollBar::handle:vertical:hover{background:#3a3a3a;}
            QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}
            QScrollBar:horizontal{background:#0a0a0a;height:10px;border:none;}
            QScrollBar::handle:horizontal{background:#2a2a2a;border-radius:5px;}

            QProgressBar{background:#141414;border:1px solid #2a2a2a;border-radius:4px;
                         text-align:center;color:#ffffff;font-weight:bold;}
            QProgressBar::chunk{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
                                 stop:0 #1a5a1a,stop:1 #3a9a3a);border-radius:3px;}

            QPushButton{background:#1b2636;color:#f3f4f6;
                        border:1px solid #2a3a50;border-radius:5px;
                        padding:5px 14px;font-weight:bold;}
            QPushButton:hover{background:#223149;border-color:#3b82f6;}
            QPushButton:pressed{background:#26364e;}
            QPushButton:disabled{background:#111827;color:#5f6b7a;border-color:#1f2937;}
            QPushButton#btn_submit{background:#17362d;color:#8ee6bc;border:1px solid #2f8f6a;}
            QPushButton#btn_submit:hover{background:#1d4438;border-color:#3fbf8f;}
            QPushButton#btn_abort{background:#3a1d24;color:#f4a6b5;border:1px solid #9f4356;}
            QPushButton#btn_abort:hover{background:#4a2530;}
            QPushButton#btn_abort:disabled{background:#111827;color:#5f6b7a;border-color:#1f2937;}

            QCheckBox{color:#ffffff;background:transparent;spacing:6px;}
            QCheckBox::indicator{width:14px;height:14px;border:1px solid #2a2a2a;
                                  border-radius:3px;background:#141414;}
            QCheckBox::indicator:checked{background:#1e4a8a;border-color:#4d9fff;}
            QCheckBox::indicator:hover{border-color:#4d9fff;}

            QMenuBar{background:#060606;color:#cccccc;border-bottom:1px solid #1a1a1a;}
            QMenuBar::item:selected{background:#141414;color:#ffffff;}
            QMenu{background:#0d0d0d;color:#ffffff;border:1px solid #2a2a2a;}
            QMenu::item{padding:5px 20px 5px 10px;}
            QMenu::item:selected{background:#1e3a5a;color:#ffffff;}
            QMenu::separator{height:1px;background:#1e1e1e;margin:3px 0;}

            QStatusBar{background:#060606;color:#777777;border-top:1px solid #141414;}
            QStatusBar QLabel{color:#777777;background:transparent;}

            QSplitter::handle{background:#1a1a1a;}
            QSplitter::handle:hover{background:#2a2a2a;}

            QToolBar{background:#0a0a0a;border-bottom:1px solid #141414;spacing:6px;padding:3px;}
        """)


def main():
    logging.basicConfig(level=logging.DEBUG,format="%(asctime)s %(levelname)-8s %(name)s — %(message)s")
    for lib in ("matplotlib","matplotlib.font_manager","matplotlib.ticker",
                "matplotlib.colorizer","PIL","ase","gemmi"):
        logging.getLogger(lib).setLevel(logging.WARNING)
    app=QApplication(sys.argv)
    app.setApplicationName("QED-DFT Studio"); app.setApplicationVersion("1.3")
    win=MainWindow(); win.show(); sys.exit(app.exec())

if __name__=="__main__":
    main()
