"""微信通话式 AI 语音面试 —— 通话组件（Python 侧）。

设计要点：
- 通话界面通过 Streamlit 官方自定义组件协议挂载
  （declare_component + path 指向 ui/call_static/index.html），
  由 Streamlit 自身服务器同源托管，iframe 无沙箱、可用麦克风/摄像头；
- 组件参数恒定（name/url/key 不变）→ 组件 id 恒定 → iframe 跨 rerun
  持久存活，麦克风授权与录音状态不因页面刷新逻辑而丢失；
- 上行事件：iframe 内 postMessage
  {isStreamlitMessage:true, type:"streamlit:setComponentValue",
   value:{event:...}, dataType:"json"}
  → Python 在组件返回值中收到 {"event":"ready"|"audio"|...}；
- 下行消息：本模块生成 <script> 注入主文档，与 iframe 完成 hello 握手后
  经 window.postMessage 投递消息；AI 面试官语音优先播放服务端合成的
  微软神经语音 MP3（edge-tts，音色自然），合成失败时退回浏览器 TTS
  （并自动选择最自然的中文音色；用户点击过「开始通话」按钮，满足
  浏览器自动播放策略），全部朗读结束后才通知 iframe 开始录音，
  避免扬声器回音被录入。
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import streamlit as st
from streamlit.components.v1 import declare_component

COMPONENT_NAME = "prismai_call"
CALL_STATIC_DIR = Path(__file__).parent / "call_static"

# iframe 会上报的事件全集（未知事件一律忽略，防伪造/防错乱）
EVENTS = {"ready", "audio", "no_speech", "hangup", "restart", "quit", "mic_error"}

_component = None


def _get_component():
    global _component
    if _component is None:
        _component = declare_component(COMPONENT_NAME, path=str(CALL_STATIC_DIR))
    return _component


def render_call():
    """渲染通话界面，返回 iframe 回传的事件 dict（本轮回传为 None）。

    返回示例：{"event": "audio", "data": "<base64 wav>", "duration_sec": 12.3}
    height 为初始帧高；参数恒定可保证组件 id 稳定、iframe 跨 rerun 持久。
    """
    return _get_component()(key="prismai_call_main", default=None, height=560)


def parse_event(value):
    """校验组件返回值；只接受白名单事件，非法/未知值一律视为 None。"""
    if not isinstance(value, dict):
        return None
    event = str(value.get("event") or "")
    if event not in EVENTS:
        return None
    return value


def decode_wav_b64(data: str) -> bytes:
    """把通话组件回传的 base64 WAV（16kHz 单声道 PCM）解码为原始字节。"""
    return base64.b64decode(data)


# 微软神经语音：比浏览器默认 TTS 音色自然得多。
# 音色跟随面试模式（温柔=晓晓女声 / 标准=云希男声 / 专业=云健男声），
# 每次合成前从 session_state 取当前通话模式，切换模式无需清缓存。
_TTS_VOICES = {
    "gentle": "zh-CN-XiaoxiaoNeural",
    "standard": "zh-CN-YunxiNeural",
    "pro": "zh-CN-YunjianNeural",
}
_TTS_VOICE = "zh-CN-XiaoxiaoNeural"


def _active_voice() -> str:
    """当前通话模式的音色；未开始通话时用默认（晓晓）。"""
    mode = str(st.session_state.get("call_mode", "standard"))
    return _TTS_VOICES.get(mode, _TTS_VOICE)
_TTS_CACHE_MAX = 120         # session_state 内按文本缓存的最大条数
_TTS_MAX_FAILS = 3           # 连续合成失败多少次后本次会话停用神经语音
_TTS_TIMEOUT_SEC = 10        # 单条合成硬超时：网络慢/卡住时最多等这么久，
                             # 超时按失败处理（退回浏览器 TTS），绝不无限阻塞广播


def _synth_batch(texts: list[str]) -> list[str | None]:
    """并发合成多条文本，返回与输入顺序一致的 data URI 列表（失败项为 None）。

    并发而非逐条顺序合成：开场白+首题两条只需一次网络往返的时间。
    已停用神经语音时直接返回全 None（前端退回浏览器 TTS）。
    """
    if st.session_state.get("_call_tts_off") or not texts:
        return [None] * len(texts)
    try:
        import asyncio

        import edge_tts

        async def _one(text: str) -> str | None:
            buf = bytearray()
            try:
                comm = edge_tts.Communicate(text, _active_voice(), rate=_TTS_RATE)
                async for chunk in comm.stream():
                    if chunk.get("type") == "audio":
                        buf.extend(chunk["data"])
            except Exception:
                return None
            if not buf:
                return None
            return "data:audio/mpeg;base64," + base64.b64encode(bytes(buf)).decode("ascii")

        async def _run_all():
            tasks = [asyncio.wait_for(_one(t), timeout=_TTS_TIMEOUT_SEC) for t in texts]
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = asyncio.run(_run_all())
    except Exception:
        return [None] * len(texts)
    return [r if isinstance(r, str) else None for r in results]


def synth_speak_items(texts) -> list[dict]:
    """把待朗读文本转成 [{text, audio}] 列表：audio 为神经语音 MP3 的 data URI。

    合成失败时该条 audio=None，前端按条目自动退回浏览器 TTS；
    结果按文本缓存于 session_state（跨 rerun 有效，重复播报不再重复合成）。
    """
    cache = st.session_state.setdefault("_call_tts_cache", {})
    voice = _active_voice()
    items = []
    missing = []
    for raw in texts or []:
        text = str(raw).strip()
        if not text:
            continue
        audio = cache.get(f"{voice}|{text}")
        if not audio:
            missing.append(text)
        items.append({"text": text, "audio": audio or None})

    if missing:
        audios = _synth_batch(missing)
        failed = 0
        for text, audio in zip(missing, audios):
            if audio:
                if len(cache) >= _TTS_CACHE_MAX:
                    try:
                        cache.pop(next(iter(cache)))
                    except (StopIteration, RuntimeError):
                        pass
                cache[f"{voice}|{text}"] = audio
                for item in items:
                    if item["text"] == text:
                        item["audio"] = audio
            else:
                failed += 1
        # 失败计数：全部失败才累计（部分失败可能是单条文本问题），
        # 连续 _TTS_MAX_FAILS 次全败后本次会话停用神经语音。
        if failed == len(missing):
            fails = int(st.session_state.get("_call_tts_fails", 0)) + 1
            st.session_state["_call_tts_fails"] = fails
            if fails >= _TTS_MAX_FAILS:
                st.session_state["_call_tts_off"] = True
        else:
            st.session_state["_call_tts_fails"] = 0
    return items


# 前端播放器核心 JS：神经语音 MP3 优先，浏览器 TTS 兜底。
# 注意：不得出现字面 "<"（DOMPurify 会把 script 当标签剥掉），
# 循环比较一律写成 length>x 形式；token 用于中断旧音频后
# 阻止其迟到回调推进新播放队列。
_PLAYER_CORE_JS = (
    "function pickVoice(){"
    "if(!('speechSynthesis' in window))return null;"
    "if(Q.voice)return Q.voice;"
    "var vs=window.speechSynthesis.getVoices()||[];"
    "var pref=['Xiaoxiao','Yunxi','Xiaoyi','Yunyang','Yunjian'];"
    "for(var k=0;pref.length>k;k++){for(var i=0;vs.length>i;i++){"
    "if(vs[i].name.indexOf(pref[k])>=0&&vs[i].lang.indexOf('zh')===0){Q.voice=vs[i];return vs[i];}}}"
    "for(var j=0;vs.length>j;j++){if(vs[j].lang.indexOf('zh')===0){Q.voice=vs[j];return vs[j];}}"
    "return null;"
    "}"
    "function stopPlay(){"
    "Q.token=(Q.token||0)+1;"
    "if(Q.playing){try{Q.playing.pause();}catch(e){}Q.playing=null;}"
    "}"
    "function playOne(t,next){"
    "var token=(Q.token||0)+1;Q.token=token;"
    "var text=(typeof t==='string')?t:((t&&t.text)||'');"
    "if(t&&t.audio&&String(t.audio).indexOf('data:')===0){"
    "var a=new Audio(t.audio);"
    "Q.playing=a;"
    "var fin=false;"
    "function go(){if(fin)return;fin=true;if(Q.token!==token)return;if(Q.playing===a)Q.playing=null;next();}"
    "a.onended=go;a.onerror=go;"
    "try{var pr=a.play();if(pr&&pr.catch){pr.catch(go);}}catch(e){go();return;}"
    "setTimeout(go,90000);"
    "return;"
    "}"
    "if(!('speechSynthesis' in window)||!text){next();return;}"
    "try{window.speechSynthesis.cancel();}catch(e){}"
    "var u=new SpeechSynthesisUtterance(text);"
    "u.lang='zh-CN';u.rate=1.0;"
    "var v=pickVoice();if(v){u.voice=v;}"
    "u.onend=function(){if(Q.token===token){next();}};"
    "u.onerror=function(){if(Q.token===token){next();}};"
    "try{window.speechSynthesis.speak(u);}catch(e){next();}"
    "}"
    "function utter(list,done){"
    "var i=0;"
    "function next(){if(i>=list.length){if(done){done();}return;}playOne(list[i++],next);}"
    "stopPlay();"
    "next();"
    "}"
)


def _escape_payload(payload: str) -> str:
    """载荷作为单引号 JS 字符串嵌入 <script>：转义 < 防提前闭合 script，
    再转义反斜杠与单引号，保证 JSON.parse 后内容原样还原。"""
    return payload.replace("<", "\\u003c").replace("\\", "\\\\").replace("'", "\\'")


def emit_voice(texts) -> None:
    """在主文档注入脚本依次朗读多段文本（神经语音 MP3 优先，浏览器 TTS 兜底）。

    供对话式面试等非通话场景复用，与 broadcast 共用同一套前端播放器；
    每次注入带自增序号，保证相同文本再次播报时脚本也会重新执行。
    """
    items = synth_speak_items(texts or [])
    if not items:
        return
    payload = _escape_payload(json.dumps({"speak": items}, ensure_ascii=False))
    seq = st.session_state.get("_voice_emit_seq", 0) + 1
    st.session_state["_voice_emit_seq"] = seq
    st.html(
        "<script>/* prismai-voice #" + str(seq) + " */"
        "(function(){"
        "var P=JSON.parse('" + payload + "');"
        "var Q=window.__prismaiVoice;"
        "if(!Q){"
        "Q={voice:null,token:0,playing:null};window.__prismaiVoice=Q;"
        "if('speechSynthesis' in window){"
        "try{window.speechSynthesis.onvoiceschanged=function(){Q.voice=null;};}catch(e){}"
        "}"
        "}"
        + _PLAYER_CORE_JS +
        "utter(P.speak.slice(),null);"
        "})();</script>",
        unsafe_allow_javascript=True,
    )


def broadcast(display: dict, speak=None, listen=None) -> None:
    """向通话 iframe 推送消息，并朗读 speak 文本（神经语音优先，TTS 兜底）。

    display: 立即发送给 iframe 的消息（含 type 字段，iframe 按 type 处理）；
    speak:   需要朗读的文本列表，全部朗读结束后把 listen 消息发给 iframe；
    listen:  朗读结束（或朗读不可用）后发送给 iframe 的消息，
             通常为 {"type":"turn","phase":"listen",...}，驱动 iframe 开始录音。
    """
    items = synth_speak_items(speak or [])
    payload = _escape_payload(
        json.dumps(
            {"display": display or {}, "speak": items, "listen": listen or None},
            ensure_ascii=False,
        )
    )
    # 每次注入的脚本体必须不同（含自增序号），否则 Streamlit 前端
    # 对相同 HTML 内容不会重新挂载，脚本不会再次执行。
    # 序号放 session_state：模块每次 rerun 会被强制重载（见 app.py 顶部），
    # 模块级变量会归零，只有 session_state 能跨 run 保持递增。
    seq = st.session_state.get("_call_bcast_seq", 0) + 1
    st.session_state["_call_bcast_seq"] = seq
    st.html(
        "<script>/* prismai-call #" + str(seq) + " */"
        "(function(){"
        "var P=JSON.parse('" + payload + "');"
        "var Q=window.__prismaiCall;"
        "if(!Q){"
        "Q={win:null,pending:[],lastSpeak:[],voice:null,token:0,playing:null};window.__prismaiCall=Q;"
        "if('speechSynthesis' in window){"
        "try{window.speechSynthesis.onvoiceschanged=function(){Q.voice=null;};}catch(e){}"
        "}"
        "window.addEventListener('message',function(ev){"
        "var d=ev.data||{};"
        "if(d.source!=='prismai-call')return;"
        "if(d.type==='hello'){"
        "Q.win=ev.source;"
        "try{ev.source.postMessage({source:'prismai-call',type:'hello-ack'},'*');}catch(e){}"
        "var p=Q.pending;Q.pending=[];"
        "for(var i=0;p.length>i;i++){try{Q.win.postMessage(p[i],'*');}catch(e){}}"
        "}else if(d.type==='replay'&&Q.lastSpeak.length){"
        "utter(Q.lastSpeak.slice(),null);"
        "}"
        "});"
        "}"
        "function send(m){"
        "try{m.source='prismai-call';}catch(e){}"
        "if(Q.win){try{Q.win.postMessage(m,'*');}catch(e){}}else{Q.pending.push(m);}"
        "}"
        + _PLAYER_CORE_JS +
        "if(P.display&&P.display.type){send(P.display);}"
        "if(P.speak&&P.speak.length){"
        "Q.lastSpeak=P.speak.slice();"
        "utter(P.speak.slice(),function(){if(P.listen&&P.listen.type){send(P.listen);}});"
        "}else if(P.listen&&P.listen.type){send(P.listen);}"
        "})();</script>",
        unsafe_allow_javascript=True,
    )
