"""PrismAI 多模态 AI 面试教练 —— Streamlit 表现层（MVP 入口）。

运行：streamlit run app.py
表现层不做任何 AI 分析，仅采集输入并渲染门面返回的结构化报告（第4周五.1约束）。
视觉风格：浅鼠尾草绿主题，白色卡片 + 绿色细边。
"""
from __future__ import annotations

import hashlib
import html
import os
import sys
import tempfile
import time
import uuid

# Streamlit 热重载只重新执行主脚本；已导入的本地模块会残留在 sys.modules，
# 修改 core/data/analyzers/ui 等目录下的文件不会生效。每次 run 前强制失效
# 项目内模块（排除 .venv 第三方包），保证 import 时始终读取磁盘最新代码。
# 强制重载会丢掉模块级缓存（如 Whisper 模型，加载成本高）：
# 失效前先暂存，等本轮 import 完成后归还给新模块。
_PRESERVE_ATTRS = {
    "analyzers.speech_analyzer": ("_MODEL_CACHE", "_T2S"),
}
_preserved = {}
for _mod_name, _attrs in _PRESERVE_ATTRS.items():
    _old = sys.modules.get(_mod_name)
    if _old is not None:
        _preserved[_mod_name] = {
            _attr: getattr(_old, _attr, None) for _attr in _attrs
        }
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
for _m, _mod in list(sys.modules.items()):
    _file = getattr(_mod, "__file__", "") or ""
    if (
        not _file
        or _m == "__main__"
        or ".venv" in _file
        or "site-packages" in _file
    ):
        continue
    try:
        if os.path.abspath(_file).startswith(_PROJECT_ROOT):
            sys.modules.pop(_m, None)
    except (ValueError, OSError):
        continue

import streamlit as st

from core.diagnostics import check_environment
from core.errors import AnalysisBusyError
from core.facade import InterviewCoachFacade
from core.input_resolver import InputValidationError
from core.settings import get_settings
from data.interview_engine import extract_role, generate_follow_up
from data.jd_profiles import PROFILE_NAMES, profile_jd_text
from data.prompts import (
    INTERVIEW_MODES,
    INTRO_QUESTION,
    closing_text,
    feedback_voice_text,
    invalid_answer_warning,
    opening_text,
    review_summary,
    scoring_prompt,
)
from data.storage import (
    list_interview_sessions,
    load_history,
    load_interview_turns,
    load_knowledge,
    load_report,
    save_history_entry,
    save_interview_turn,
    save_knowledge,
)
from ui.call_component import (
    broadcast,
    decode_wav_b64,
    parse_event,
    render_call,
    synth_speak_items,
)
from ui.login import render_login

# 归还上一轮暂存的模块级缓存（见文件顶部失效逻辑）
for _mod_name, _values in _preserved.items():
    _target = sys.modules.get(_mod_name)
    if _target is not None:
        for _attr, _value in _values.items():
            if _value:
                setattr(_target, _attr, _value)

# 通话/多轮面试的业务状态键（不含控件 widget 键，退出登录与超时时清理）
_CALL_STATE_KEYS = (
    "call_user", "call_session_id", "call_turns", "call_question",
    "call_limit", "call_persist", "call_active", "call_started",
    "call_opening_done", "call_notice", "call_last_sig",
    "call_pending_broadcast", "_call_bcast_seq",
    "call_mode", "call_role", "call_intro_retried",
)


@st.cache_data(ttl=30, show_spinner=False)
def _system_diagnostics():
    return check_environment()

# ---------------------------------------------------------------------------
# 全局样式
# ---------------------------------------------------------------------------
st.set_page_config(page_title="PrismAI 面试教练", page_icon="🎯", layout="wide")

CSS = """
<style>
/* 基础 */
:root {
  --bg:        #eef2ee;
  --card:      #ffffff;
  --green:     #5a9b7e;
  --green-2:   #4a8c6e;
  --green-bg:  #e8f3ee;
  --green-line:#b7d4c4;
  --red:       #d6606b;
  --text:      #2d3436;
  --muted:     #6b7280;
  --border:    #e0e8e0;
  --radius:    12px;
}
[data-testid="stAppViewContainer"] { background: var(--bg); }
[data-testid="stSidebar"]          { background: #ffffff; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] *        { color: var(--text); }
h1, h2, h3, h4 { color: var(--text) !important; font-family: "PingFang SC","Microsoft YaHei",sans-serif; }

/* 顶部主标题 */
.hero-title {
  font-size: 2.2rem; font-weight: 700; color: var(--text);
  text-align: center; margin: 0.4rem 0 0.1rem; letter-spacing: 1px;
}
.hero-sub {
  text-align: center; color: var(--muted); font-size: 0.95rem; margin-bottom: 1.2rem;
}
.hero-underline {
  width: 90px; height: 4px; margin: 0 auto 1.6rem;
  background: linear-gradient(90deg, var(--green) 0%, var(--green-2) 100%);
  border-radius: 2px;
}

/* 卡片通用 */
.prism-card {
  background: var(--card); border-radius: var(--radius);
  box-shadow: 0 2px 10px rgba(0,0,0,0.05);
  border: 1px solid var(--border);
  padding: 1.1rem 1.3rem; margin-bottom: 1rem;
}
.card-title { font-size: 1.15rem; font-weight: 700; color: var(--text); margin-bottom: 0.6rem; }
.card-sub   { font-size: 0.85rem; color: var(--muted); line-height: 1.55; }

/* 侧栏 */
.side-brand { font-size: 1.05rem; font-weight: 700; color: var(--text); margin-bottom: 0.9rem; }
.side-label { font-size: 0.8rem; color: var(--muted); font-weight: 600; margin-bottom: 0.25rem; }
.side-user  { font-size: 0.98rem; font-weight: 700; margin-bottom: 0.9rem; }
.stat-grid { display: flex; gap: 0.55rem; margin-bottom: 1rem; }
.stat-box {
  flex: 1; background: var(--card); border: 1px solid var(--border);
  border-radius: 10px; padding: 0.6rem 0.3rem; text-align: center;
}
.stat-box .num { font-size: 1.1rem; font-weight: 700; color: var(--green-2); line-height: 1.2; }
.stat-box .lbl { font-size: 0.7rem; color: var(--muted); margin-top: 0.15rem; }

/* 题目展示卡 */
.question-card {
  background: var(--green-bg); border-left: 4px solid var(--green);
  border-radius: var(--radius); padding: 1rem 1.2rem; margin: 0.6rem 0 1rem;
}
.question-card .q-tag {
  display: inline-block; background: var(--green); color: #fff;
  font-size: 0.72rem; padding: 2px 10px; border-radius: 10px; margin-right: 8px;
  vertical-align: middle;
}
.question-card .q-text { font-size: 1rem; color: var(--text); font-weight: 500; line-height:1.6; vertical-align: middle; }
.question-card .q-index {
  float: right; font-size: 0.78rem; color: var(--muted);
  padding-top: 4px;
}

/* 题目选择 radio 菜单 */
[data-testid="stRadio"] {
  width: 100% !important;
}
[data-testid="stRadio"] > div {
  background: var(--card);
  border-radius: var(--radius);
  border: 1px solid var(--border);
  padding: 0.5rem;
  max-height: 260px;
  overflow-y: auto;
  width: 100% !important;
}
[data-testid="stRadio"] [role="radiogroup"] {
  width: 100% !important;
}
[data-testid="stRadio"] label {
  padding: 0.7rem 0.9rem !important;
  border-radius: 8px !important;
  margin-bottom: 4px !important;
  transition: background .15s;
  cursor: pointer;
}
[data-testid="stRadio"] label:hover {
  background: var(--green-bg) !important;
}
[data-testid="stRadio"] label:has(input:checked) {
  background: var(--green-bg) !important;
  border-left: 3px solid var(--green);
}
[data-testid="stRadio"] label span {
  color: var(--text) !important;
  font-size: 0.92rem !important;
}

/* 评分卡 */
.score-big {
  font-size: 2.6rem; font-weight: 800; color: var(--green-2);
  line-height: 1; letter-spacing: -1px;
}
.score-label { font-size: 0.85rem; color: var(--muted); margin-top: 0.3rem; }
.dim-row { display:flex; justify-content:space-between; align-items:center;
           padding: 0.55rem 0; border-bottom: 1px solid var(--border); }
.dim-row:last-child { border-bottom:none; }
.dim-name { font-size: 0.92rem; color: var(--text); font-weight: 500; }
.dim-val  { font-size: 1rem;  font-weight: 700; color: var(--green-2); }
.badge    { font-size: 0.7rem; padding: 1px 8px; border-radius: 8px; margin-left:6px; }
.badge-ok  { background: var(--green-bg); color: var(--green-2); }
.badge-deg { background: #fdf0e6; color: #c97a2b; }

/* 建议 */
.sug-item {
  background: var(--green-bg); border-radius: 8px;
  padding: 0.6rem 0.9rem; margin-bottom: 0.5rem;
  font-size: 0.9rem; color: var(--text); line-height:1.5;
  border-left: 3px solid var(--green);
}

/* 面试模式 pills 美化（Streamlit 1.66 渲染为 stButtonGroup + role=radio） */
[data-testid="stButtonGroup"] button[role="radio"] {
  border-radius: 20px !important;
  font-size: 0.9rem !important;
  padding: 0.3rem 1.1rem !important;
  border: 1px solid var(--border) !important;
  background: var(--card) !important;
  color: var(--muted) !important;
}
[data-testid="stButtonGroup"] button[role="radio"][aria-checked="true"] {
  background: var(--green) !important;
  color: #fff !important;
  border-color: var(--green) !important;
}

/* 录音控件美化 */
[data-testid="stAudioInput"] button {
  background: var(--green) !important; color: #fff !important;
  border: none !important; border-radius: 10px !important;
  min-height: 52px !important; font-weight: 600 !important; font-size: 1rem !important;
  width: 100% !important;
}
[data-testid="stAudioInput"] button:hover { background: var(--green-2) !important; }
[data-testid="stAudioInput"] [data-testid="stAudio"] {
  border: 1px solid var(--border) !important;
  border-radius: 10px !important; padding: 0.4rem !important;
  background: var(--card) !important;
}

/* 按钮 */
.stButton > button {
  background: var(--green) !important; color: #fff !important;
  border: none !important; border-radius: 8px !important;
  font-weight: 600 !important; padding: 0.5rem 1.6rem !important;
  transition: background .2s;
}
.stButton > button:hover { background: var(--green-2) !important; }
.stButton > button:disabled { background: #c8d8d0 !important; color: #8aa098 !important; }

/* selectbox 容器 */
[data-testid="stSelectbox"] > div > div {
  background: var(--card) !important; border: 1px solid var(--border) !important;
  border-radius: 8px;
}

/* spinner 文字 */
.stSpinner > div { color: var(--green-2); }

/* 主内容区加宽：让 stMain 占满侧边栏以外的全部空间 */
[data-testid="stMain"] {
  flex: 1 1 0% !important;
  width: 100% !important;
  max-width: none !important;
}
[data-testid="stMainBlockContainer"],
.block-container {
  max-width: 100% !important;
  width: 100% !important;
  padding-left: 2rem !important;
  padding-right: 2rem !important;
}
/* 题目选择区按钮卡片 */
.question-picker button {
  height: auto !important;
  min-height: 64px !important;
  text-align: left !important;
  justify-content: flex-start !important;
  padding: 0.8rem 1rem !important;
  border-radius: 10px !important;
  font-size: 0.92rem !important;
  line-height: 1.45 !important;
  white-space: normal !important;
}
.question-picker button[data-testid="stBaseButton-secondary"] {
  background: var(--card) !important;
  color: var(--text) !important;
  border: 1.5px solid var(--border) !important;
}
.question-picker button[data-testid="stBaseButton-secondary"]:hover {
  border-color: var(--green) !important;
  background: var(--green-bg) !important;
}
.question-picker button[data-testid="stBaseButton-primary"] {
  background: var(--green) !important;
  color: #fff !important;
  border: 1.5px solid var(--green) !important;
}
.cat-picker button {
  border-radius: 20px !important;
  font-size: 0.85rem !important;
  height: 36px !important;
}
.cat-picker button[data-testid="stBaseButton-secondary"] {
  background: var(--card) !important;
  color: var(--muted) !important;
  border: 1px solid var(--border) !important;
}
.cat-picker button[data-testid="stBaseButton-primary"] {
  background: var(--green) !important;
  color: #fff !important;
  border: 1px solid var(--green) !important;
}

/* 标题下划线（参考图的绿色短下划线） */
.section-head {
  font-size: 1.18rem; font-weight: 700; color: var(--text);
  padding-bottom: 0.4rem; margin: 1.5rem 0 0.9rem;
  border-bottom: 2px solid var(--green);
  display: inline-block; letter-spacing: 0.5px;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 顶部标题
# ---------------------------------------------------------------------------
st.markdown('<div class="hero-title">PrismAI AI 面试教练</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">语音作答 · 即时评分 · AI 追问 · 深度复盘</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-underline"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 登录校验：未登录时只渲染登录/注册页
# ---------------------------------------------------------------------------
auth_user = st.session_state.get("auth_user")
if auth_user:
    idle_timeout = int(get_settings()["auth"]["idle_timeout_sec"])
    last_active = float(st.session_state.get("auth_last_active", 0) or 0)
    if idle_timeout and last_active and time.time() - last_active > idle_timeout:
        for key in (
            "auth_user", "auth_last_active", "report", "report_key",
            "history_report", "history_report_sid", "history_report_missing",
        ):
            st.session_state.pop(key, None)
        for key in _CALL_STATE_KEYS:
            st.session_state.pop(key, None)
        st.rerun()
    st.session_state["auth_last_active"] = time.time()
if not auth_user:
    render_login()

# ---------------------------------------------------------------------------
# 左侧功能导航（仿参考图左侧卡片）
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="side-brand">🎯 PrismAI 面试教练</div>', unsafe_allow_html=True
    )
    try:
        _hist_stats = load_history(username=auth_user)
    except Exception:
        _hist_stats = []
    if _hist_stats:
        _stat_n = len(_hist_stats)
        _stat_avg = round(sum(float(h["overall"]) for h in _hist_stats) / _stat_n, 1)
        _stat_best = max(float(h["overall"]) for h in _hist_stats)
        st.markdown(
            f'<div class="stat-grid">'
            f'<div class="stat-box"><div class="num">{_stat_n}</div>'
            f'<div class="lbl">练习次数</div></div>'
            f'<div class="stat-box"><div class="num">{_stat_avg}</div>'
            f'<div class="lbl">平均得分</div></div>'
            f'<div class="stat-box"><div class="num">{_stat_best:.0f}</div>'
            f'<div class="lbl">最高得分</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="side-label">当前用户</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="side-user">{html.escape(str(auth_user))}</div>',
        unsafe_allow_html=True,
    )
    if st.button("退出登录", key="logout", use_container_width=True):
        for key in (
            "auth_user", "auth_last_active", "report", "report_key",
            "history_report", "history_report_sid", "history_report_missing",
        ):
            st.session_state.pop(key, None)
        for key in _CALL_STATE_KEYS:
            st.session_state.pop(key, None)
        st.rerun()
    with st.expander("系统状态"):
        for item in _system_diagnostics():
            icon = {"ok": "✅", "warn": "⚠️", "error": "❌"}.get(
                item.get("status"), "•"
            )
            st.markdown(
                f"{icon} **{item.get('name', '')}**：{item.get('detail', '')}"
            )

# ---------------------------------------------------------------------------
# 主体：面试发起卡（面试设置移至主页面）
# ---------------------------------------------------------------------------
knowledge = load_knowledge(str(auth_user)) if auth_user else {}
knowledge_prefix = f"knowledge_{auth_user}"
# ---------------------------------------------------------------------------
# 📞 微信通话式 AI 语音面试
# ---------------------------------------------------------------------------

def _call_context(knowledge):
    return {
        key: knowledge.get(key, "")
        for key in ("candidate_profile", "project_evidence", "job_description")
        if knowledge.get(key)
    }


def _call_question_msg(q, round_no, limit, feedback="", hint=""):
    """下发「AI 提问」消息：iframe 展示题目/反馈并进入播报动画。"""
    return {
        "type": "turn",
        "phase": "speak",
        "round": round_no,
        "total": limit,
        "question": q.get("content", ""),
        "feedback": feedback,
        "hint": hint,
    }


def _call_listen_msg(q, round_no, limit):
    """TTS 播报结束后下发的「请回答」消息：iframe 打开麦克风开始录音。"""
    return {
        "type": "turn",
        "phase": "listen",
        "round": round_no,
        "total": limit,
        "question": q.get("content", ""),
    }


def _call_save_review(auth_user, turns, knowledge_prefix):
    """通话结束：把整场复盘沉淀追加进知识库（与多轮面试区逻辑一致）。"""
    try:
        summary = review_summary(turns)
        kb = load_knowledge(str(auth_user))
        previous = str(kb.get("interview_review", "")).strip()
        merged = f"{summary}\n\n{previous}" if previous else summary
        kb["interview_review"] = merged[:8000]
        save_knowledge(str(auth_user), kb)
        st.session_state[f"{knowledge_prefix}_review_draft"] = merged[:8000]
        st.session_state[f"{knowledge_prefix}_review_ver"] = (
            st.session_state.get(f"{knowledge_prefix}_review_ver", 0) + 1
        )
    except (OSError, ValueError):
        pass


def _call_avg_dims(turns):
    """整场通话的内容/语音平均分（写入历史趋势记录）。"""
    dims = {}
    for key in ("content", "speech"):
        vals = [
            float(t["report"].dimension_scores.get(key) or 0.0)
            for t in turns
        ]
        dims[key] = round(sum(vals) / len(vals), 1) if vals else 0.0
    return dims


def _call_end(turns, closing, auth_user, knowledge_prefix):
    """结束本场通话：沉淀复盘、写入历史趋势、下发结束广播（平均分）。"""
    if turns and st.session_state.get("call_persist"):
        _call_save_review(auth_user, turns, knowledge_prefix)
    if turns:
        avg = round(
            sum(float(t["report"].overall_score) for t in turns) / len(turns), 1
        )
        try:
            save_history_entry(
                str(auth_user),
                str(st.session_state.get("call_session_id", "call")),
                "通话面试",
                avg,
                _call_avg_dims(turns),
            )
        except (OSError, ValueError):
            pass
    else:
        avg = None
    synth_speak_items([closing])
    _defer_broadcast(
        {"type": "end", "closing": closing, "avg": avg, "total": len(turns)},
        speak=[closing],
        listen=None,
    )


def _defer_broadcast(display, speak=None, listen=None):
    """把下行广播暂存到 session_state，交由下一次 rerun 开头渲染。

    事件处理紧跟着 st.rerun()：同一次 run 内调用的 st.html 增量会被
    Streamlit 丢弃（实测 DOM 中不出现），因此广播必须延迟到下一个
    完整 run（该 run 正常结束、增量正常刷新）再注入。
    """
    st.session_state["call_pending_broadcast"] = {
        "display": display,
        "speak": speak or [],
        "listen": listen,
    }


def _handle_call_event(ev, auth_user, knowledge, knowledge_prefix, first_question):
    """处理通话 iframe 回传事件，推进对话状态并广播下一轮消息。"""
    kind = ev.get("event") or ""
    q = st.session_state.get("call_question") or {}
    turns = list(st.session_state.get("call_turns", []))
    limit = int(st.session_state.get("call_limit") or 4)
    round_no = len(turns) + 1
    session_id = st.session_state.get("call_session_id", "call")
    q_text = q.get("content", "")
    st.session_state["call_notice"] = None

    if kind == "ready":
        # 首次接听 → 开场白 + 第一题；中途重连（如麦克风恢复）→ 重播当前题
        st.session_state["call_started"] = True
        speak = []
        if not st.session_state.get("call_opening_done"):
            speak.append(opening_text(limit, q_text))
            st.session_state["call_opening_done"] = True
        speak.append(q_text)
        _defer_broadcast(
            _call_question_msg(q, round_no, limit),
            speak=speak,
            listen=_call_listen_msg(q, round_no, limit),
        )
        return

    if kind == "audio":
        try:
            wav_bytes = decode_wav_b64(str(ev.get("data") or ""))
        except (ValueError, TypeError):
            st.session_state["call_notice"] = "⚠️ 音频数据接收失败，已请求重新作答。"
            _defer_broadcast(
                _call_question_msg(q, round_no, limit, hint="音频接收失败，请再回答一次。"),
                speak=["音频接收失败，请再回答一次。", q_text],
                listen=_call_listen_msg(q, round_no, limit),
            )
            return
        fd, audio_path = tempfile.mkstemp(suffix=".wav")
        with os.fdopen(fd, "wb") as f:
            f.write(wav_bytes)
        report = None
        context = _call_context(knowledge)
        try:
            with st.spinner("正在转写并分析你的回答…"):
                report = InterviewCoachFacade().run_interview(
                    question=q,
                    audio_path=audio_path,
                    video_path=None,
                    job_category=q.get("category", "default"),
                    username=auth_user,
                    candidate_context=context,
                )
        except (InputValidationError, AnalysisBusyError) as exc:
            st.session_state["call_notice"] = f"⚠️ {exc}"
        except Exception as exc:
            st.session_state["call_notice"] = f"本轮处理失败：{exc}"
        finally:
            try:
                os.remove(audio_path)
            except OSError:
                pass
        if report is None:
            # 分析失败：原题重答，不消耗轮次
            _defer_broadcast(
                _call_question_msg(q, round_no, limit, hint="本轮分析遇到问题，请再回答一次。"),
                speak=["本轮分析遇到问题，请再回答一次。", q_text],
                listen=_call_listen_msg(q, round_no, limit),
            )
            return
        mode = str(st.session_state.get("call_mode", "standard"))
        mode_cfg = INTERVIEW_MODES.get(mode, INTERVIEW_MODES["standard"])
        # 无效作答闸门（乱回答/答非所问/机械重复）
        is_gate = bool(
            report.transcript
            and getattr(report, "scoring_details", {}).get("method") == "gate"
        )
        # 开场轮（自我介绍+应聘岗位）：一次无效作答先温和重问一次，
        # 不消耗轮次；再次无效才按态度问题严肃终止。
        if is_gate and round_no == 1 and not st.session_state.get("call_intro_retried"):
            st.session_state["call_intro_retried"] = True
            hint = (
                "我没有听清你的自我介绍。请更完整地介绍一下自己："
                "先说明你要应聘的岗位，再讲一两条相关的经历或优势。"
            )
            _defer_broadcast(
                _call_question_msg(q, round_no, limit, hint=hint),
                speak=[hint, q_text],
                listen=_call_listen_msg(q, round_no, limit),
            )
            return
        turn = {
            "question": q_text,
            "transcript": report.transcript,
            "report": report,
        }
        turns.append(turn)
        st.session_state["call_turns"] = turns
        if st.session_state.get("call_persist"):
            save_interview_turn(
                session_id, round_no, auth_user, q_text, turn["transcript"], report
            )
        if is_gate:
            # 严肃提醒态度问题，不再追问下一题，直接结束本场通话
            _call_end(turns, invalid_answer_warning(), auth_user, knowledge_prefix)
            return
        if round_no >= limit:
            _call_end(turns, closing_text(turns), auth_user, knowledge_prefix)
            return
        # 开场轮回答后提取应聘岗位：后续追问围绕该岗位动态展开
        if round_no == 1:
            role = extract_role(turn["transcript"])
            st.session_state["call_role"] = role
        follow_up_context = dict(context)
        if knowledge.get("interviewer_style"):
            follow_up_context["interviewer_style"] = (
                mode_cfg["style"] + "补充：" + knowledge["interviewer_style"]
            )
        else:
            follow_up_context["interviewer_style"] = mode_cfg["style"]
        role = st.session_state.get("call_role") or ""
        if role:
            jd = follow_up_context.get("job_description", "")
            follow_up_context["job_description"] = (
                (jd + "\n") if jd else ""
            ) + f"面试者应聘的岗位：{role}"
        follow_up = generate_follow_up(
            q,
            report.transcript,
            candidate_context=follow_up_context,
            prior_turns=turns,
            report=report,
        )
        new_q = {
            "id": f"C{round_no + 1}",
            "category": q.get("category", "综合"),
            "content": follow_up["question"],
        }
        st.session_state["call_question"] = new_q
        feedback_text = feedback_voice_text(report, mode=mode)
        # 「AI 正在思考」期间预合成反馈与追问语音，广播时直接命中缓存
        synth_speak_items([feedback_text, new_q["content"]])
        _defer_broadcast(
            _call_question_msg(
                new_q, round_no + 1, limit, feedback=feedback_text
            ),
            speak=[feedback_text, new_q["content"]],
            listen=_call_listen_msg(new_q, round_no + 1, limit),
        )
        return

    if kind == "no_speech":
        # 静音/回答过短：重播同题，不消耗轮次；语气随面试模式调整
        mode = str(st.session_state.get("call_mode", "standard"))
        if mode == "gentle":
            hint = "没关系，不用紧张。我没有听清，请再慢慢说一次。"
        elif mode == "pro":
            hint = "请集中注意力，直接、清晰地回答问题，再试一次。"
        else:
            hint = "我没有听清你的回答，请再回答一次。"
        _defer_broadcast(
            _call_question_msg(q, round_no, limit, hint=hint),
            speak=[hint, q_text],
            listen=_call_listen_msg(q, round_no, limit),
        )
        return

    if kind == "hangup":
        if turns:
            _call_end(turns, closing_text(turns), auth_user, knowledge_prefix)
        else:
            _defer_broadcast(
                {"type": "end", "closing": "本次通话未开始正式问答。", "avg": None, "total": 0},
                speak=None,
                listen=None,
            )
        return

    if kind == "restart":
        # 再来一场：重置会话，iframe 回到响铃界面等待再次接听
        st.session_state["call_session_id"] = uuid.uuid4().hex[:12]
        st.session_state["call_turns"] = []
        st.session_state["call_question"] = dict(first_question)
        st.session_state["call_started"] = False
        st.session_state["call_opening_done"] = False
        st.session_state["call_notice"] = None
        st.session_state["call_role"] = None
        st.session_state["call_intro_retried"] = False
        _defer_broadcast({"type": "reset"}, speak=None, listen=None)
        return

    if kind == "quit":
        st.session_state["call_active"] = False
        st.session_state["call_started"] = False
        st.session_state["call_notice"] = None
        return

    if kind == "mic_error":
        st.session_state["call_notice"] = (
            "麦克风不可用：" + str(ev.get("message") or "请检查浏览器权限后重试。")
        )


st.markdown('<div class="section-head">📞 微信通话式 AI 语音面试</div>', unsafe_allow_html=True)
st.caption(
    "接听来电后，面试官会先问你应聘的岗位，之后由 AI 根据你的回答"
    "动态提问和评分；说完自动停止录音，全程免手动操作。"
)

if st.session_state.get("call_user") != auth_user:
    for key in _CALL_STATE_KEYS:
        st.session_state.pop(key, None)
    st.session_state["call_user"] = auth_user

call_turns = st.session_state.get("call_turns", [])
call_active = bool(st.session_state.get("call_active"))

if not call_active:
    # 面试模式：只影响提问风格与语气，不改变评分标准
    call_mode = st.pills(
        "面试模式",
        options=list(INTERVIEW_MODES),
        selection_mode="single",
        format_func=lambda m: INTERVIEW_MODES[m]["label"],
        default="standard",
        key="call_mode_picker",
    )
    call_mode = call_mode or "standard"
    st.caption(INTERVIEW_MODES[call_mode]["desc"])

    _call_col1, _call_col2 = st.columns([1, 2.4])
    with _call_col1:
        call_limit = st.slider("通话轮数", 2, 6, 4, key="call_round_limit")
    with _call_col2:
        call_persist = st.checkbox(
            "将通话问答转写和评分保存到本机知识数据库（不保存音视频）",
            value=True,
            key=f"call_persist_{auth_user}",
        )

    start_col, _ = st.columns([1, 3])
    with start_col:
        if st.button("📞 开始通话", type="primary", key="call_start_btn"):
            st.session_state["call_session_id"] = uuid.uuid4().hex[:12]
            st.session_state["call_turns"] = []
            st.session_state["call_question"] = dict(INTRO_QUESTION)
            st.session_state["call_limit"] = call_limit
            st.session_state["call_persist"] = call_persist
            st.session_state["call_mode"] = call_mode
            st.session_state["call_role"] = None
            st.session_state["call_intro_retried"] = False
            st.session_state["call_active"] = True
            st.session_state["call_started"] = False
            st.session_state["call_opening_done"] = False
            st.session_state["call_notice"] = None
            st.session_state["call_last_sig"] = None
            # 点「开始通话」时就把开场白与首题的语音合成好并缓存：
            # 响铃界面出现后，一接听就能秒连面试官（否则接听瞬间的
            # 广播要现场合成语音，网络慢时表现为“一直连接不上”）。
            with st.spinner("正在准备面试官语音…"):
                synth_speak_items([
                    opening_text(call_limit, INTRO_QUESTION.get("content", "")),
                    INTRO_QUESTION.get("content", ""),
                ])
            st.rerun()
else:
    # 上一轮事件处理暂存的广播：在本 run 开头渲染（本 run 正常结束，
    # 增量不会被 st.rerun() 丢弃），再继续处理组件新回传
    pending_b = st.session_state.pop("call_pending_broadcast", None)
    if pending_b:
        broadcast(
            pending_b["display"],
            speak=pending_b["speak"],
            listen=pending_b["listen"],
        )
    st.caption(
        f"通话中：{st.session_state.get('call_session_id', '')}｜"
        f"已完成 {len(call_turns)}/{st.session_state.get('call_limit', 4)} 轮"
    )
    ev = parse_event(render_call())
    if ev is None and not st.session_state.get("call_started"):
        # 接听前预热开场白与首题的神经语音：合成的 MP3 缓存进 session_state，
        # 接听后广播直接命中缓存、秒连面试官。edge-tts 走微软网络，慢时
        # 曾让广播阻塞十几秒，表现为“一直连接不上”。
        # 未命中缓存的文本为空串会被跳过；已缓存时此调用零网络开销。
        warm_q = (st.session_state.get("call_question") or {}).get("content", "")
        warm_limit = int(st.session_state.get("call_limit") or 4)
        synth_speak_items([opening_text(warm_limit, warm_q), warm_q])
    if ev:
        # 组件回传值在 rerun 间持久保留，用事件序号去重：
        # 每个事件只处理一次，重复 rerun 不会再次广播/推进状态
        sig = f"{ev.get('event')}:{ev.get('seq')}"
        if sig != st.session_state.get("call_last_sig"):
            st.session_state["call_last_sig"] = sig
            _handle_call_event(
                ev, auth_user, knowledge, knowledge_prefix, INTRO_QUESTION
            )
            st.rerun()
    if st.session_state.get("call_notice"):
        st.warning(st.session_state["call_notice"])
    if st.button("退出通话模式", key="call_exit_btn"):
        st.session_state["call_active"] = False
        st.session_state["call_started"] = False
        st.session_state["call_notice"] = None
        st.rerun()

if call_turns:
    if not call_active:
        saved_note = (
            "本场问答与评分已保存到本机知识库（可在下方「查看已保存的通话记录」中打开）。"
            if st.session_state.get("call_persist")
            else "本场未开启保存，记录仅保留在当前页面。"
        )
        st.caption(f"本次通话共 {len(call_turns)} 轮。{saved_note}")
    for turn_index, turn in enumerate(call_turns, start=1):
        with st.chat_message("assistant"):
            st.markdown(f"**第 {turn_index} 题**")
            st.write(turn["question"])
        with st.chat_message("user"):
            st.write(turn.get("transcript") or "未能获得语音转写。")
            st.metric("本轮能力评分", f"{turn['report'].overall_score}/100")
            with st.expander(f"第 {turn_index} 轮反馈与证据"):
                for suggestion in turn["report"].suggestions:
                    st.markdown(f"- {suggestion}")
                st.json(turn["report"].scoring_details)


saved_dialogs = list_interview_sessions(auth_user)
if saved_dialogs:
    with st.expander("查看已保存的通话记录"):
        dialog_options = {
            item["session_id"]: (
                f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(item['created_at']))}"
                f" · {item['turn_count']} 轮 · 均分 {item['average_score']}"
            )
            for item in saved_dialogs
        }
        selected_dialog = st.selectbox(
            "选择历史通话",
            list(dialog_options),
            format_func=lambda sid: dialog_options[sid],
            key="saved_dialog_select",
        )
        for saved_turn in load_interview_turns(selected_dialog, auth_user):
            st.markdown(f"**題目：** {saved_turn['question']}")
            st.write(saved_turn["transcript"] or "无可用转写")
            st.caption(f"本轮评分：{saved_turn['overall']}/100")

# ---------------------------------------------------------------------------
# 历史趋势
# ---------------------------------------------------------------------------
history = load_history(username=auth_user)
if history:
    st.markdown('<div class="section-head">历史练习趋势</div>', unsafe_allow_html=True)
    st.line_chart({"综合评分": [h["overall"] for h in history]})

    with st.expander("查看历史报告详情"):
        options = {
            h["session_id"]: (
                f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(h['created_at']))}"
                f" · {h['overall']} 分"
            )
            for h in reversed(history)
        }
        selected = st.selectbox(
            "选择一次练习",
            list(options.keys()),
            format_func=lambda sid: options[sid],
            key="history_select",
        )
        if st.button("加载所选报告", key="history_load"):
            payload = load_report(selected, username=auth_user)
            st.session_state["history_report"] = payload
            st.session_state["history_report_sid"] = selected
            st.session_state["history_report_missing"] = payload is None

        if st.session_state.get("history_report_sid") == selected:
            payload = st.session_state.get("history_report")
            if payload:
                question_text = payload.get("question", {}).get("content", "")
                if question_text:
                    st.markdown(f"**题目：** {question_text}")
                c1, c2 = st.columns(2)
                c1.metric("综合评分", payload.get("overall_score", "—"))
                confidence = float(payload.get("confidence", 0.0) or 0.0)
                c2.metric("评分置信度", f"{int(round(confidence * 100))}%")
                labels = {
                    "content": "内容质量",
                    "speech": "表达能力",
                    "vision": "精神状态",
                }
                for key, score in payload.get("dimension_scores", {}).items():
                    st.write(f"{labels.get(key, key)}：{score}")
                for suggestion in payload.get("suggestions", [])[:8]:
                    st.markdown(f"- {suggestion}")
                with st.expander("完整报告 JSON"):
                    st.json(payload)
            elif st.session_state.get("history_report_missing"):
                st.warning("该记录生成于完整报告持久化之前，暂无报告 JSON。")
        elif st.session_state.get("history_report_sid"):
            st.caption("点击“加载所选报告”查看详情。")

# ---------------------------------------------------------------------------
# ⚙️ 高级设置：知识库与评分标准
# ---------------------------------------------------------------------------
st.markdown('<div class="section-head">⚙️ 高级设置</div>', unsafe_allow_html=True)

with st.expander("我的面试知识库与 AI 提示词", expanded=False):
    st.caption(
        "资料按当前账号保存在本机 SQLite 中，仅用于生成追问和核对回答；"
        "面试官风格不会改变固定评分标准。"
    )
    with st.form(f"knowledge_form_{auth_user}"):
        candidate_profile = st.text_area(
            "个人背景与核心技能",
            value=knowledge.get("candidate_profile", ""),
            key=f"{knowledge_prefix}_profile",
            height=90,
            placeholder="填写经历、技能、职责范围；避免填写身份证号、密码等敏感信息。",
        )
        project_evidence = st.text_area(
            "项目事实 / 可核验成果",
            value=knowledge.get("project_evidence", ""),
            key=f"{knowledge_prefix}_projects",
            height=110,
            placeholder="按项目记录：目标、你的职责、采取的行动、数据结果与证据来源。",
        )
        jd_ver = st.session_state.get(f"{knowledge_prefix}_jd_ver", 0)
        jd_draft = st.session_state.get(f"{knowledge_prefix}_jd_draft")
        job_description = st.text_area(
            "目标岗位 JD / 面试要求",
            value=(jd_draft if jd_draft is not None
                   else knowledge.get("job_description", "")),
            key=f"{knowledge_prefix}_jd_v{jd_ver}",
            height=100,
        )
        profile_row = st.columns([2, 1])
        with profile_row[0]:
            jd_profile = st.selectbox(
                "从内置岗位画像导入 JD",
                ["— 选择岗位 —"] + PROFILE_NAMES,
                key=f"{knowledge_prefix}_jd_profile",
                help="内置 10 个常见岗位的职责、技能与 JD 模板；导入后可再修改。",
            )
        with profile_row[1]:
            st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
            import_jd = st.form_submit_button("导入到 JD")
        interviewer_style = st.text_input(
            "面试官风格（可选）",
            value=knowledge.get("interviewer_style", ""),
            key=f"{knowledge_prefix}_style",
            max_chars=160,
            placeholder="例如：友善但会追问数据和技术取舍",
        )
        review_ver = st.session_state.get(f"{knowledge_prefix}_review_ver", 0)
        review_draft = st.session_state.get(f"{knowledge_prefix}_review_draft")
        interview_review = st.text_area(
            "面试复盘沉淀（每场多轮面试结束后自动追加，可修改）",
            value=(review_draft if review_draft is not None
                   else knowledge.get("interview_review", "")),
            key=f"{knowledge_prefix}_review_v{review_ver}",
            height=120,
            placeholder="完成多轮面试后自动沉淀：每轮得分与最需提升的点。",
        )
        knowledge_saved = st.form_submit_button("保存知识库")
    if import_jd and jd_profile and jd_profile != "— 选择岗位 —":
        st.session_state[f"{knowledge_prefix}_jd_draft"] = profile_jd_text(jd_profile)
        st.session_state[f"{knowledge_prefix}_jd_ver"] = (
            st.session_state.get(f"{knowledge_prefix}_jd_ver", 0) + 1
        )
        st.rerun()
    if knowledge_saved:
        try:
            save_knowledge(str(auth_user), {
                "candidate_profile": candidate_profile,
                "project_evidence": project_evidence,
                "job_description": job_description,
                "interviewer_style": interviewer_style,
                "interview_review": interview_review,
            })
            knowledge = load_knowledge(str(auth_user))
            st.session_state[f"{knowledge_prefix}_jd_draft"] = None
            st.session_state[f"{knowledge_prefix}_review_draft"] = None
            st.success("知识库已保存，后续单题评估和模拟面试会使用这些资料。")
        except (OSError, ValueError) as exc:
            st.error(f"保存知识库失败：{exc}")

with st.expander("评分量表与 AI 评估提示词"):
    st.markdown(
        "**总分 = 内容质量 × 55% + 语音表达 × 45%。**"
    )
    st.markdown(
        "**量化评分规则（rubric 2.0）：** 内容五维各设 5 个可观察指标，"
        "每项 20 分，命中即得分。规则评分器与 LLM 使用同一指标清单，"
        "保证两条评分路径共用一把尺子。"
    )
    st.markdown(
        "- 逻辑性 22%：回应题意 / 观点明确 / 因果依据 / 分步论证 / 结论呼应\n"
        "- 深度 24%：量化结果 / 个人贡献 / 情境背景 / 技术业务细节 / 约束取舍\n"
        "- 结构 20%：开头总起 / 分层展开 / 过程完整 / 结果导向 / 收尾总结\n"
        "- 表达 14%：句段适中 / 篇幅适中 / 无冗余重复 / 用词具体 / 衔接自然\n"
        "- 准确度 20%：紧扣题意 / 与资料一致 / 数据可核验 / 无过度断言 / 标注不确定性"
    )
    st.caption(
        "分档锚点：命中 0 项缺失/离题；1 项笼统；2 项有内容但依据不足；"
        "3 项基本达标；4 项有具体证据；5 项证据充分并能解释取舍。"
        "准确度无可核验资料时固定 50 分，不臆测事实。"
    )
    st.code(scoring_prompt(), language="text")
