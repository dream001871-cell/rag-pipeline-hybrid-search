"""RAG 中文检索与问答界面，通过 HTTP 调用后端服务。"""

import os

import streamlit as st
import requests

DEFAULT_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
MODES = {
    "混合检索（向量 + 关键词）": "hybrid",
    "仅向量检索": "dense",
    "仅关键词检索（BM25）": "sparse",
}


def api_get(base: str, path: str, timeout: int = 5):
    try:
        return requests.get(f"{base}{path}", timeout=timeout)
    except requests.RequestException:
        return None


def api_post(base: str, path: str, payload: dict, timeout: int = 30):
    try:
        resp = requests.post(f"{base}{path}", json=payload, timeout=timeout)
    except requests.RequestException as exc:
        return None, None, f"无法连接后端： {exc.__class__.__name__}"
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", "")
        except Exception:
            detail = resp.text[:200]
        return None, resp.status_code, detail
    return resp.json(), resp.status_code, None


def fetch_ask(base: str, question: str, mode: str, top_k: int):
    return api_post(base, "/v1/ask", {
        "question": question,
        "retrieval_mode": mode,
        "top_k": top_k,
    }, timeout=180)


def fetch_retrieve(base: str, question: str, mode: str, top_k: int):
    return api_post(base, "/v1/retrieve", {
        "question": question,
        "mode": mode,
        "top_k": top_k,
    })


def fmt(v) -> str:
    if v is None:
        return "—"
    return f"{v:.4f}"


def render_chunk(item: dict, rank: int, highlight: bool = False):
    """展示来源片段，并高亮用户选中的引用。"""
    meta_lines = []
    if item.get("source_file"):
        meta_lines.append(f"**{item['source_file'].split('/')[-1]}**")
    if item.get("section"):
        meta_lines.append(f"§ {item['section']}")
    if item.get("chunking_strategy"):
        strategy = item["chunking_strategy"]
        strategy_label = {"fixed": "固定长度", "recursive": "递归切分", "semantic": "语义切分"}.get(strategy, strategy)
        meta_lines.append(f"切分策略={strategy_label}")
    header = f"**#{rank}** · {' · '.join(meta_lines)}"
    scores = (
        f"重排 **{fmt(item.get('rerank_score'))}** · "
        f"RRF 融合 **{fmt(item.get('rrf_score'))}** · "
        f"向量 **{fmt(item.get('dense_score'))}** · "
        f"关键词 **{fmt(item.get('sparse_score'))}**"
    )
    if highlight:
        st.markdown(f"### {header}")
        st.markdown(f"`{scores}` — 此片段被答案引用")
        st.info(item.get("content", ""))
    else:
        st.markdown(f"### {header}")
        st.markdown(f"`{scores}`")
        st.markdown(item.get("content", ""))


def render_confidence(conf: dict):
    if not conf:
        st.write("本次响应没有可用的置信度信息。")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("综合置信度", f"{conf.get('composite', 0):.2%}")
    c2.metric("检索相关度", f"{conf.get('retrieval_confidence', 0):.2%}")
    c3.metric("引用覆盖率", f"{conf.get('citation_coverage', 0):.2%}")
    c4.metric("回答完整度", f"{conf.get('completeness', 0):.2%}")
    w = conf.get("weights")
    if w:
        st.caption(
            "权重："
            f"检索 {w.get('retrieval')}, "
            f"引用 {w.get('citation')}, "
            f"完整度 {w.get('completeness')}"
        )


def render_citation_buttons(citations: dict, sources: list):
    """用引用编号定位对应的原文片段。"""
    if not citations or not citations.get("citations"):
        st.markdown("本次回答没有引用。")
        return
    cols = st.columns(min(len(citations["citations"]), 6))
    for col, cit in zip(cols, citations["citations"]):
        n = cit["citation"]  # 引用编号，例如 "[1]"
        if col.button(
            n,
            key=f"cite_{n}",
            help=(cit.get("claim") or cit.get("reason") or "")[:120],
        ):
            st.session_state["selected_citation"] = n
    if st.session_state.get("selected_citation"):
        st.caption("正在高亮所选引用对应的原文片段。")


def render_answer(body: dict):
    if not body:
        return
    st.subheader("回答")
    if body.get("can_answer"):
        st.markdown(body["answer"])
        st.markdown("---")
        with st.expander("引用来源 — 点击编号查看对应原文", expanded=True):
            render_citation_buttons(body.get("citations"), body.get("sources") or [])
        st.subheader("置信度")
        render_confidence(body.get("confidence"))
    else:
        declined = body.get("declined") or {}
        reason = declined.get("reason", "置信度不足")
        if reason == "insufficient_confidence":
            reason = "当前证据不足，暂时无法给出可靠回答"
        st.warning(f"暂时无法回答：{reason}。")
        if declined.get("summary"):
            st.write(declined["summary"])
        st.subheader("原因说明")
        render_confidence(body.get("confidence"))
        if declined.get("found"):
            st.markdown("找到的部分相关片段：")
            for f in declined["found"]:
                st.markdown(f"- {f}")


def render_ranking(base: str, question: str, top_k: int, mode: str = "hybrid"):
    st.subheader("按相关度排序的检索片段")
    hybrid, _, _ = fetch_retrieve(base, question, mode, top_k)
    if not hybrid:
        st.warning("暂时无法检索，请检查后端服务是否已启动。")
        return
    select = st.session_state.get("selected_citation")
    for rank, item in enumerate(hybrid["ranking"], start=1):
        citation_target = select == f"[{rank}]"
        with st.expander(
            f"#{rank} · {item.get('source_file','').split('/')[-1]} · "
            f"重排 {fmt(item.get('rerank_score'))}",
            expanded=citation_target,
        ):
            render_chunk(item, rank, highlight=citation_target)


def render_comparison(base: str, question: str, top_k: int):
    st.subheader("混合检索与纯向量检索对比")
    hybrid, hs, hdetail = fetch_retrieve(base, question, "hybrid", top_k)
    dense, ds, ddetail = fetch_retrieve(base, question, "dense", top_k)
    if not hybrid and not dense:
        st.warning("暂时无法对比，请检查后端连接。")
        return
    if hs == 503 or ds == 503:
        st.info("后端暂不可用，请先配置有效的模型服务密钥，或开启纯检索模式。")
    if hybrid is None:
        st.error(hdetail)
        return
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 混合检索")
        if not hybrid["ranking"]:
            st.write("没有检索结果")
        dense_ids = {item["chunk_id"] for item in dense["ranking"]} if dense else set()
        for rank, item in enumerate(hybrid["ranking"], start=1):
            tag = "两种检索均命中" if item["chunk_id"] in dense_ids else ""
            st.markdown(
                f"**#{rank}** 重排 **{fmt(item.get('rerank_score'))}** {tag}"
            )
            st.caption(item.get("content", "")[:220])
    with c2:
        st.markdown("### 仅向量检索")
        if not dense or not dense["ranking"]:
            st.write("没有检索结果")
        hybrid_ids = {item["chunk_id"] for item in hybrid["ranking"]} if hybrid else set()
        for rank, item in enumerate(dense["ranking"], start=1):
            tag = "两种检索均命中" if item["chunk_id"] in hybrid_ids else ""
            st.markdown(
                f"**#{rank}** 重排 **{fmt(item.get('rerank_score'))}** {tag}"
            )
            st.caption(item.get("content", "")[:220])


def render():
    st.set_page_config(
        page_title="RAG 混合检索工作台",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("RAG 混合检索工作台")
    st.caption(
        "查看混合检索、向量检索和关键词检索的结果；配置模型后可生成带引用的回答，并检查置信度。"
    )

    with st.sidebar:
        st.header("服务连接")
        base = st.text_input("后端服务地址", value=DEFAULT_BASE_URL)
        try:
            health = api_get(base, "/healthz")
            if health is not None and health.ok:
                hb = health.json()
                st.success(f"服务正常，已索引 {hb['vector_chunks']} 个片段")
            else:
                st.warning("无法连接后端服务")
        except Exception:
            st.warning("无法连接后端服务")

        try:
            docs = api_get(base, "/v1/documents")
            if docs is not None and docs.ok and docs.json():
                st.caption(f"已索引 {len(docs.json())} 条文档记录")
        except Exception:
            pass

        st.divider()
        st.header("检索与问答")
        retrieval_only = st.toggle(
            "纯检索模式（无需密钥）",
            value=os.environ.get("RAG_RETRIEVAL_ONLY", "false").lower() == "true",
        )
        mode_label = st.radio(
            "检索方式",
            list(MODES.keys()),
            index=0,
        )
        retrieval_mode = MODES[mode_label]
        top_k = st.slider("返回片段数（Top-K）", min_value=3, max_value=20, value=20)
        show_compare = st.toggle(
            "显示混合检索与纯向量检索对比", value=True
        )

    question = st.text_input("输入你的问题", placeholder="例如：What is FastAPI?（当前样例资料为英文）")
    action_label = "开始检索" if retrieval_only else "生成回答"
    asked = st.button(action_label, type="primary")

    if not question.strip():
        st.info(f"请在上方输入问题，然后点击 **{action_label}**。")
        return

    if asked or st.session_state.get("last_question") == question:
        st.session_state["last_question"] = question
        st.session_state.pop("selected_citation", None)

        # 没有密钥时直接走真实检索，不调用需要模型服务的问答接口。
        if retrieval_only:
            render_ranking(base, question, top_k, retrieval_mode)
            if show_compare:
                render_comparison(base, question, top_k)
            return

        with st.spinner("正在检索并生成回答……"):
            body, code, detail = fetch_ask(base, question, retrieval_mode, top_k)
        if body:
            render_answer(body)
        else:
            st.error(f"问答请求失败（状态码 {code}）：{detail}")
            if code == 503:
                st.warning(
                    "请在项目 .env 中填写 DEEPSEEK_API_KEY 并重启服务；未配置密钥时可继续使用纯检索模式。"
                )

        st.markdown("---")
        render_ranking(base, question, top_k, retrieval_mode)

        if show_compare:
            st.markdown("---")
            render_comparison(base, question, top_k)


if __name__ == "__main__":
    render()
