import streamlit as st
import pandas as pd
import numpy as np
import os, glob, io, re
from datetime import datetime

st.set_page_config(page_title="Billing Aggregator — онлайн сводка", layout="wide")

WANTED_ORDER = [
    "Арутюнян Артур Арамович",
    "Белозёров Кирилл Сергеевич",
    "Блинова Марина Владимировна",
    "Блиновская Алена Игоревна",
    "Бурмистров Егор Евгеньевич",
    "Бухтов Александр Владимирович",
    "Горелов Алексей Владимирович",
    "Дмитриев Олег Викторович",
    "Долгих Егор Владимирович",
    "Капилети Екатерина Олеговна",
    "Кашин Александр Владимирович",
    "Конакова Дарья Дмитриевна",
    "Корнилов Никита Дмитриевич",
    "Кузнецов Станислав Алексеевич",
    "Курмаков Динар Рамдисович",
    "Курочкин Станислав Евгеньевич",
    "Маслова Анастасия Сергеевна",
    "Новикова Ольга Дмитриевна",
    "Носова Алена Сергеевна",
    "Орлова Яна Юрьевна",
    "Павлова Дарья Олеговна",
    "Панкова Вероника Викторовна",
    "Прекрасная Елена Андреевна",
    "Садовая Виктория Павловна",
    "Сорокина Ольга Олеговна",
    "Сухов Дмитрий Андреевич",
    "Тавадян Вероника Суреновна",
    "Шуварикова Лариса Александровна",
    "Южанина Екатерина Алексеевна",
]

DEF_COLS = [
    "ticket","ticket_full","priority","summary","status",
    "created_ts","updated_ts","assignee","tags","difficulty",
    "weight","importance","weight_tarif","importance_",
    "hero_total","resizes_total","total","utm_term"
]
CANON_MAP = {
    "ticket": ["ticket","ключ","тикет","key","task","тип"],
    "ticket_full": ["ticket_full","ссылка","url","link","ticket url"],
    "priority": ["priority","приоритет"],
    "summary": ["summary","задача","описание","task_summary"],
    "status": ["status","статус"],
    "created_ts": ["created_ts","дата создания","создано","created"],
    "updated_ts": ["updated_ts","обновлено","updated"],
    "assignee": ["assignee","исполнитель","сотрудник","user","owner"],
    "tags": ["tags","теги","utm-term","utm_term"],
    "difficulty": ["difficulty","сложность"],
    "weight": ["weight","вес"],
    "importance": ["importance","важность"],
    "weight_tarif": ["weight_tarif","тариф","тариф_вес","тариф по весу","тариф по задачам"],
    "importance_": ["importance_","важность_","важность тариф"],
    "hero_total": ["hero_total","геро баннер","геро","hero"],
    "resizes_total": ["resizes_total","ресайзы","resizes","resize_total","resizes_total"],
    "total": ["total","итого","сумма","amount"],
    "utm_term": ["utm_term","utm-term","метка","tag"]
}

def canonize_columns(df):
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    rename = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        mapped = None
        for canon, variants in CANON_MAP.items():
            if lc == canon or lc in variants:
                mapped = canon
                break
        if mapped:
            rename[c] = mapped
    df = df.rename(columns=rename)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    return df

def _score_header_row(df_raw, row_idx):
    headers = df_raw.iloc[row_idx].astype(str).str.strip().str.lower().tolist()
    synonyms = sum(CANON_MAP.values(), []) + list(CANON_MAP.keys())
    return sum(1 for h in headers if h in synonyms)

def _read_excel_smart(bdata):
    df_raw = pd.read_excel(bdata, header=None)
    best_header, best_score = 0, -1
    for i in range(min(8, len(df_raw))):
        sc = _score_header_row(df_raw, i)
        if sc > best_score:
            best_header, best_score = i, sc
    df = pd.read_excel(bdata, header=best_header)
    if len(df):
        first_vals = df.iloc[0].astype(str).str.lower().tolist()
        type_tokens = {"string","int64","float64","object","datetime64[ns]"}
        if first_vals and all((v in type_tokens or v=="") for v in first_vals):
            df = df.iloc[1:].reset_index(drop=True)
    return df.loc[:, ~df.columns.duplicated()]

def detect_name_login(s):
    if pd.isna(s):
        return None, None
    s = str(s).strip()
    m = re.search(r"^(.*)\(([^)]+)\)\s*$", s)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    parts = s.split("|")
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return s.strip(), None

def clean_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = re.sub(r"\(.*?\)", "", name)
    name = name.split("|")[0]
    return re.sub(r"\s+", " ", name).strip()

def load_employee_file_from_bytes(name, data):
    try:
        bio = io.BytesIO(data)
        df = _read_excel_smart(bio)
    except Exception as e:
        raise RuntimeError(f"Не удалось прочитать файл {name}: {e}")
    df = canonize_columns(df)
    keep = [c for c in DEF_COLS if c in df.columns]
    if keep:
        df = df[keep].copy()
    df['source_file'] = name
    if 'ticket' in df.columns:
        df = df[~df['ticket'].astype(str).str.lower().str.startswith(('общая сумма','итого'))]
    if 'assignee' in df.columns:
        pairs = df['assignee'].apply(detect_name_login)
        df['assignee_name'] = pairs.apply(lambda x: clean_name(x[0]))
    else:
        df['assignee_name'] = ""
    if 'total' in df.columns:
        df['total'] = pd.to_numeric(df['total'], errors='coerce').fillna(0.0)
    else:
        df['total'] = 0.0
    return df

def load_employee_file_from_path(path):
    with open(path, 'rb') as f:
        return load_employee_file_from_bytes(os.path.basename(path), f.read())

st.header("Загрузка файлов")
col1, col2 = st.columns(2)
files_data = []

with col1:
    mode = st.radio("Источник", ["Папка на диске", "Загрузить файлы вручную"])
    if mode == "Папка на диске":
        folder = st.text_input("Путь к папке с Excel (*.xlsx)", value="./input")
        if st.button("Просканировать папку"):
            st.session_state['paths'] = sorted(glob.glob(os.path.join(folder, "*.xlsx")))
        for p in st.session_state.get('paths', []):
            files_data.append(load_employee_file_from_path(p))
        if 'paths' in st.session_state:
            st.caption(f"Найдено файлов: {len(st.session_state['paths'])}")
    else:
        uploads = st.file_uploader("Загрузите Excel-отчёты (*.xlsx)", type=["xlsx"], accept_multiple_files=True)
        if uploads:
            for up in uploads:
                files_data.append(load_employee_file_from_bytes(up.name, up.read()))

with col2:
    ndfl_rate = st.number_input("Ставка НДФЛ (0.13 = 13%)", min_value=0.0, max_value=0.99, value=0.13, step=0.01)
    period = st.text_input("Период (подпись)", value=datetime.now().strftime("%Y-%m %d–%d"))
    st.caption("Онлайн-сводка: до загрузки — нули, затем суммы появляются. Gross-up = base/(1-rate).")

def build_summary(files_data, rate):
    rows = [{"Сотрудник": name, "Сумма из отчёта": 0.0, "Сумма с НДФЛ": 0.0} for name in WANTED_ORDER]
    base_df = pd.DataFrame(rows)
    if not files_data:
        return base_df
    all_tasks = pd.concat(files_data, ignore_index=True)
    if 'assignee_name' not in all_tasks:
        all_tasks['assignee_name'] = ""
    if 'total' not in all_tasks:
        all_tasks['total'] = 0.0
    all_tasks['assignee_name'] = all_tasks['assignee_name'].apply(clean_name)
    by = all_tasks.groupby('assignee_name', dropna=False)['total'].sum().reset_index()
    by.rename(columns={'assignee_name':'Сотрудник','total':'Сумма из отчёта'}, inplace=True)
    m = base_df.merge(by, on="Сотрудник", how="left", suffixes=("","_new"))
    m["Сумма из отчёта"] = m["Сумма из отчёта_new"].fillna(m["Сумма из отчёта"]).fillna(0.0)
    if "Сумма из отчёта_new" in m.columns:
        m.drop(columns=["Сумма из отчёта_new"], inplace=True)
    m["Сумма с НДФЛ"] = (m["Сумма из отчёта"] / (1 - rate)).round(2)
    m["Сумма из отчёта"] = m["Сумма из отчёта"].round(2)
    others = by[~by["Сотрудник"].isin(m["Сотрудник"])].copy()
    if len(others):
        others["Сумма с НДФЛ"] = (others["Сумма из отчёта"] / (1 - rate)).round(2)
        others["Сумма из отчёта"] = others["Сумма из отчёта"].round(2)
        m = pd.concat([m, others], ignore_index=True)
    return m

summary_df = build_summary(files_data, ndfl_rate)

if "custom_table" not in st.session_state:
    st.session_state["custom_table"] = summary_df.copy()
else:
    old = st.session_state["custom_table"]
    merged = old.merge(summary_df, on="Сотрудник", how="outer", suffixes=("","_new"))
    for col in ["Сумма из отчёта","Сумма с НДФЛ"]:
        merged[col] = merged.get(f"{col}_new", pd.Series(dtype=float)).fillna(merged[col]).fillna(0.0)
        if f"{col}_new" in merged.columns:
            merged.drop(columns=[f"{col}_new"], inplace=True)
    existing_names = list(old["Сотрудник"])
    new_names = [n for n in list(summary_df["Сотрудник"]) if n not in existing_names]
    ordered_names = existing_names + new_names
    merged["__order"] = merged["Сотрудник"].apply(lambda x: ordered_names.index(x) if x in ordered_names else 10**6)
    merged = merged.sort_values("__order").drop(columns=["__order"]).reset_index(drop=True)
    st.session_state["custom_table"] = merged

st.header("Список сотрудников (онлайн)")
st.caption("Таблица всегда видна: изначально нули, затем суммы появляются при загрузке файлов.")

edited = st.data_editor(
    st.session_state["custom_table"],
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Сотрудник": st.column_config.TextColumn("Сотрудник", required=False),
        "Сумма из отчёта": st.column_config.NumberColumn("Сумма из отчёта", disabled=True, format="%.2f"),
        "Сумма с НДФЛ": st.column_config.NumberColumn("Сумма с НДФЛ", disabled=True, format="%.2f"),
    },
    hide_index=True,
    key="editor_table_live",
)

st.session_state["custom_table"]["Сотрудник"] = edited["Сотрудник"]

st.subheader("Управление порядком / составом")
c1, c2, c3, c4 = st.columns([2,1,1,2])
names_for_select = list(st.session_state["custom_table"]["Сотрудник"])
with c1:
    sel_name = st.selectbox("Выбрать строку", names_for_select if names_for_select else ["—"])
with c2:
    if st.button("↑ Вверх") and names_for_select and sel_name in names_for_select:
        df = st.session_state["custom_table"]
        idx = df.index[df["Сотрудник"]==sel_name][0]
        if idx > 0:
            df.iloc[[idx-1, idx]] = df.iloc[[idx, idx-1]].values
            st.session_state["custom_table"] = df.reset_index(drop=True)
with c3:
    if st.button("↓ Вниз") and names_for_select and sel_name in names_for_select:
        df = st.session_state["custom_table"]
        idx = df.index[df["Сотрудник"]==sel_name][0]
        if idx < len(df)-1:
            df.iloc[[idx, idx+1]] = df.iloc[[idx+1, idx]].values
            st.session_state["custom_table"] = df.reset_index(drop=True)
with c4:
    if st.button("Удалить выбранную") and names_for_select and sel_name in names_for_select:
        df = st.session_state["custom_table"]
        st.session_state["custom_table"] = df[df["Сотрудник"]!=sel_name].reset_index(drop=True)

st.markdown("— или —")
a1, a2 = st.columns([3,1])
with a1:
    new_name = st.text_input("Добавить строку (ФИО, можно пусто)")
with a2:
    if st.button("Добавить в конец"):
        df = st.session_state["custom_table"]
        st.session_state["custom_table"] = pd.concat([df, pd.DataFrame([{
            "Сотрудник": new_name.strip() if new_name else "",
            "Сумма из отчёта": 0.0,
            "Сумма с НДФЛ": 0.0,
        }])], ignore_index=True)

st.markdown("---")
st.subheader("Скачать отчёт (Excel)")
if st.button("Сформировать общий отчет"):
    import io
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        if files_data:
            all_tasks = pd.concat(files_data, ignore_index=True)
            all_tasks.to_excel(writer, sheet_name="All tasks", index=False)
        st.session_state["custom_table"].to_excel(writer, sheet_name="For accounting", index=False)
    st.download_button(
        "Скачать billing_summary.xlsx",
        data=output.getvalue(),
        file_name="billing_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
