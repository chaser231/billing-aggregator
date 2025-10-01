import streamlit as st
import pandas as pd
import numpy as np
import os, glob, io, re, zipfile
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
    "weight","importance","weight_tariff","importance_tariff",
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
    "weight_tariff": ["weight_tarif","weight_tariff","тариф","тариф_вес","тариф по весу","тариф по задачам"],
    "importance_tariff": ["importance_tariff","importance_tarif","importance_","важность_","важность тариф","тариф важность","тариф_важность","тариф по важности"],
    "hero_total": ["hero_total","геро баннер","геро","hero","стоимость хиро","всего стоимость хиро","итог по хиро","хиро итого"],
    "resizes_total": ["resizes_total","ресайзы","resizes","resize_total","resizes_total","стоимость ресайзов","всего стоимость ресайзов","итог по ресайзам","ресайзы итого"],
    "total": ["total","итого","сумма","amount","итого по тикету","всего"],
    "utm_term": ["utm_term","utm-term","utm-метка","метка","tag","utm_метка","utm метка"]
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
    
    # Добавляем отсутствующие колонки из DEF_COLS с пустыми значениями
    numeric_cols = ['weight', 'importance', 'weight_tariff', 'importance_tariff', 'hero_total', 'resizes_total', 'total', 'difficulty']
    for col in DEF_COLS:
        if col not in df.columns:
            if col in numeric_cols:
                df[col] = 0.0  # Числовые колонки инициализируем нулями
            else:
                df[col] = ""   # Текстовые колонки пустой строкой
    
    # Выбираем нужные колонки в правильном порядке
    df = df[DEF_COLS].copy()
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
    """Загружает файл Excel сотрудника с обработкой ошибок"""
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл не найден: {path}")
        
        if not path.lower().endswith('.xlsx'):
            raise ValueError(f"Неподдерживаемый формат файла: {path}")
            
        with open(path, 'rb') as f:
            return load_employee_file_from_bytes(os.path.basename(path), f.read())
            
    except Exception as e:
        # Возвращаем пустой DataFrame с информацией об ошибке
        error_df = pd.DataFrame(columns=['ticket', 'summary', 'total', 'assignee_name', 'source_file'])
        error_df['source_file'] = [f"ERROR: {os.path.basename(path)}"]
        error_df['assignee_name'] = [f"Ошибка загрузки: {str(e)}"]
        error_df['total'] = [0.0]
        return error_df

def validate_ticket_unified(row, tolerance=0.01):
    """
    Унифицированная валидация тикета по трем правилам:
    1. weight × weight_tariff = hero_total
    2. importance × importance_tariff = resizes_total  
    3. hero_total + resizes_total = total
    
    Args:
        row: строка данных
        tolerance: допустимое отклонение
        
    Returns:
        {
            'is_valid': bool,
            'corrections': [{'field': str, 'old_value': float, 'new_value': float, 'reason': str}],
            'total_diff': float
        }
    """
    # Извлекаем и конвертируем значения
    weight = pd.to_numeric(row.get('weight', 0), errors='coerce') or 0
    importance = pd.to_numeric(row.get('importance', 0), errors='coerce') or 0
    weight_tariff = pd.to_numeric(row.get('weight_tariff', 0), errors='coerce') or 0
    importance_tariff = pd.to_numeric(row.get('importance_tariff', 0), errors='coerce') or 0
    hero_total = pd.to_numeric(row.get('hero_total', 0), errors='coerce') or 0
    resizes_total = pd.to_numeric(row.get('resizes_total', 0), errors='coerce') or 0
    total = pd.to_numeric(row.get('total', 0), errors='coerce') or 0
    
    corrections = []
    total_diff = 0.0
    
    # Правило 1: weight × weight_tariff = hero_total
    expected_hero = round(weight * weight_tariff, 2)
    if abs(expected_hero - hero_total) > tolerance:
        corrections.append({
            'field': 'hero_total',
            'old_value': hero_total,
            'new_value': expected_hero,
            'reason': f'weight({weight}) × weight_tariff({weight_tariff}) = {expected_hero}'
        })
        total_diff += abs(expected_hero - hero_total)
        hero_total = expected_hero  # Обновляем для следующих расчетов
    
    # Правило 2: importance × importance_tariff = resizes_total
    expected_resizes = round(importance * importance_tariff, 2)
    if abs(expected_resizes - resizes_total) > tolerance:
        corrections.append({
            'field': 'resizes_total',
            'old_value': resizes_total,
            'new_value': expected_resizes,
            'reason': f'importance({importance}) × importance_tariff({importance_tariff}) = {expected_resizes}'
        })
        total_diff += abs(expected_resizes - resizes_total)
        resizes_total = expected_resizes  # Обновляем для следующих расчетов
    
    # Правило 3: hero_total + resizes_total = total
    expected_total = round(hero_total + resizes_total, 2)
    if abs(expected_total - total) > tolerance:
        corrections.append({
            'field': 'total',
            'old_value': total,
            'new_value': expected_total,
            'reason': f'hero_total({hero_total}) + resizes_total({resizes_total}) = {expected_total}'
        })
        total_diff += abs(expected_total - total)
    
    return {
        'is_valid': len(corrections) == 0,
        'corrections': corrections,
        'total_diff': round(total_diff, 2)
    }


def validate_employee_report(employee_data, tolerance=0.01):
    """
    Валидация отчета одного сотрудника:
    Проверяет, что сумма (hero_total + resizes_total) всех строк = общий итог сотрудника
    
    Args:
        employee_data: DataFrame с данными сотрудника
        tolerance: допустимое отклонение
        
    Returns:
        {
            'is_valid': bool,
            'employee_name': str,
            'calculated_total': float,
            'reported_total': float,  
            'difference': float,
            'task_errors': [{...}]  # Ошибки на уровне тикетов
        }
    """
    # Убираем строки-итоги
    clean_data = employee_data.copy()
    if 'ticket' in clean_data.columns:
        mask = ~clean_data['ticket'].astype(str).str.lower().str.contains(
            'итого|общая сумма|total|сумма:', na=False
        )
        clean_data = clean_data[mask]
    
    # Валидируем каждый тикет и собираем исправления
    task_errors = []
    corrected_data = clean_data.copy()
    
    for idx, row in clean_data.iterrows():
        ticket_validation = validate_ticket_unified(row, tolerance)
        if not ticket_validation['is_valid']:
            task_errors.append({
                'row_idx': idx,
                'ticket': str(row.get('ticket', 'N/A'))[:30],
                'corrections': ticket_validation['corrections'],
                'total_diff': ticket_validation['total_diff']
            })
            
            # Применяем исправления к данным
            for correction in ticket_validation['corrections']:
                corrected_data.loc[idx, correction['field']] = correction['new_value']
    
    # Считаем итоги после исправлений
    hero_sum = pd.to_numeric(corrected_data.get('hero_total', 0), errors='coerce').fillna(0).sum()
    resizes_sum = pd.to_numeric(corrected_data.get('resizes_total', 0), errors='coerce').fillna(0).sum()
    calculated_total = round(hero_sum + resizes_sum, 2)
    
    # Получаем заявленный итог из исходных данных
    total_sum = pd.to_numeric(clean_data.get('total', 0), errors='coerce').fillna(0).sum()
    
    # Определяем имя сотрудника
    employee_name = clean_data.get('assignee_name', clean_data.get('assignee', ['Unknown'])).iloc[0] if len(clean_data) > 0 else 'Unknown'
    
    difference = round(abs(calculated_total - total_sum), 2)
    
    return {
        'is_valid': difference <= tolerance and len(task_errors) == 0,
        'employee_name': str(employee_name),
        'calculated_total': calculated_total,
        'reported_total': round(total_sum, 2),
        'difference': difference,
        'task_errors': task_errors,
        'corrected_data': corrected_data
    }

# 🔧 АВТОНОМНЫЙ ИНСТРУМЕНТ ДЛЯ ПРОВЕРКИ ОТДЕЛЬНОГО ОТЧЕТА
st.markdown("## 🔧 Проверка отдельного отчета")
st.markdown("*Независимый инструмент для валидации и исправления одного файла*")

with st.expander("📄 Загрузить и проверить отчет сотрудника", expanded=False):
    st.caption("Загрузите Excel файл с отчетом сотрудника для проверки и автоматического исправления ошибок")
    
    # Загрузка файла
    uploaded_single_file = st.file_uploader(
        "Выберите файл отчета (.xlsx)", 
        type=['xlsx'], 
        key="single_report_uploader",
        help="Загрузите Excel файл с отчетом одного сотрудника"
    )
    
    if uploaded_single_file is not None:
        try:
            # Загружаем и обрабатываем файл
            single_file_data = load_employee_file_from_bytes(uploaded_single_file.name, uploaded_single_file.read())
            
            if len(single_file_data) > 0:
                st.success(f"✅ Файл загружен: **{uploaded_single_file.name}**")
                
                # Показываем краткую информацию
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Строк задач", len(single_file_data))
                with col2:
                    employee_name = single_file_data['assignee_name'].iloc[0] if len(single_file_data) > 0 else "Unknown"
                    st.metric("Сотрудник", employee_name)
                with col3:
                    total_sum = pd.to_numeric(single_file_data['total'], errors='coerce').fillna(0).sum()
                    st.metric("Общая сумма", f"{total_sum:.2f} ₽")
                
                # Кнопка валидации
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if st.button("🔍 Проверить отчет", key="validate_single_report"):
                        # Запускаем валидацию
                        validation_result = validate_employee_report(single_file_data, tolerance=0.01)
                        
                        # Сохраняем результат
                        st.session_state['standalone_validation_result'] = {
                            'result': validation_result,
                            'file_name': uploaded_single_file.name,
                            'original_data': single_file_data
                        }
                        
                        if validation_result['is_valid']:
                            st.success("🎉 Отчет корректен! Ошибок не найдено.")
                        else:
                            st.error(f"❌ Найдены ошибки в отчете")
                            
                            # Показываем статистику ошибок
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Ошибок в тикетах", len(validation_result['task_errors']))
                            with col2:
                                st.metric("Исправленная сумма", f"{validation_result['calculated_total']:.2f} ₽")
                            with col3:
                                st.metric("Разница", f"{validation_result['difference']:.2f} ₽")
                
                # Показываем результаты валидации, если есть
                if 'standalone_validation_result' in st.session_state:
                    result_data = st.session_state['standalone_validation_result']
                    validation_result = result_data['result']
                    
                    if not validation_result['is_valid'] and validation_result['task_errors']:
                        with st.expander(f"🔧 Детали ошибок ({len(validation_result['task_errors'])})", expanded=True):
                            for i, error in enumerate(validation_result['task_errors'][:10]):
                                st.write(f"**Строка {error['row_idx']}**: {error['ticket']}")
                                for correction in error['corrections']:
                                    st.write(f"  • **{correction['field']}**: {correction['old_value']} → {correction['new_value']}")
                                    st.write(f"    *{correction['reason']}*")
                                if i < len(validation_result['task_errors'][:10]) - 1:
                                    st.write("")
                            
                            if len(validation_result['task_errors']) > 10:
                                st.caption(f"... и еще {len(validation_result['task_errors']) - 10} ошибок")
                        
                        # Кнопка экспорта исправленного файла
                        with col2:
                            if st.button("💾 Скачать исправленный отчет", key="download_corrected_single"):
                                corrected_data = validation_result['corrected_data']
                                
                                # Фильтруем колонки для экспорта
                                export_data = corrected_data[DEF_COLS].copy()
                                
                                # Создаем Excel файл
                                output = io.BytesIO()
                                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                    export_data.to_excel(writer, sheet_name="Corrected Report", index=False)
                                
                                st.download_button(
                                    label="📥 Скачать исправленный файл",
                                    data=output.getvalue(),
                                    file_name=f"corrected_{uploaded_single_file.name}",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    help="Скачать Excel файл с исправленными данными для отправки сотруднику",
                                    key="download_corrected_single_final"
                                )
            else:
                st.warning("⚠️ Файл пустой или не содержит данных")
                
        except Exception as e:
            st.error(f"❌ Ошибка при обработке файла: {str(e)}")

st.markdown("---")

st.header("Загрузка файлов")
col1, col2 = st.columns(2)
files_data = []

with col1:
    mode = st.radio("Источник", ["Папка на диске", "Загрузить файлы вручную"])
    if mode == "Папка на диске":
        # Инициализируем текущую папку в session_state
        if 'current_folder' not in st.session_state:
            st.session_state['current_folder'] = os.path.abspath("./input")
        
        # Показываем текущую папку и кнопку для ручного ввода
        col_path, col_browse = st.columns([3, 1])
        with col_path:
            manual_folder = st.text_input("Путь к папке с Excel (*.xlsx)", value=st.session_state['current_folder'])
            if manual_folder != st.session_state['current_folder']:
                st.session_state['current_folder'] = manual_folder
        
        with col_browse:
            if st.button("📁 Выбрать папку"):
                st.session_state['show_browser'] = True
        
        # Файловый браузер
        if st.session_state.get('show_browser', False):
            st.markdown("**Выберите папку:**")
            current_dir = st.session_state['current_folder']
            
            # Показываем текущий путь
            st.code(f"📂 {current_dir}")
            
            # Быстрые ссылки на популярные папки
            st.markdown("**Быстрый доступ:**")
            quick_links = [
                ("🏠 Домашняя папка", os.path.expanduser("~")),
                ("📄 Документы", os.path.expanduser("~/Documents")),
                ("⬇️ Загрузки", os.path.expanduser("~/Downloads")),
                ("🖥️ Рабочий стол", os.path.expanduser("~/Desktop")),
            ]
            
            cols = st.columns(len(quick_links))
            for i, (name, path) in enumerate(quick_links):
                with cols[i]:
                    if os.path.exists(path) and st.button(name, key=f"quick_{i}", use_container_width=True):
                        st.session_state['current_folder'] = path
                        st.rerun()
            
            try:
                # Кнопка "Вверх" (родительская папка)
                parent_dir = os.path.dirname(current_dir)
                if parent_dir != current_dir:  # Не корневая папка
                    if st.button("⬆️ Вверх (родительская папка)"):
                        st.session_state['current_folder'] = parent_dir
                        st.rerun()
                
                # Список папок в текущей директории
                if os.path.exists(current_dir) and os.path.isdir(current_dir):
                    items = []
                    try:
                        for item in os.listdir(current_dir):
                            if item.startswith('.'):  # Пропускаем скрытые папки
                                continue
                            item_path = os.path.join(current_dir, item)
                            if os.path.isdir(item_path):
                                items.append(("📁", item, item_path))
                    except PermissionError:
                        st.warning("Нет доступа к этой папке")
                    
                    # Сортируем папки по алфавиту
                    items.sort(key=lambda x: x[1].lower())
                    
                    # Показываем папки в контейнере с прокруткой
                    if items:
                        st.markdown(f"**Папки ({len(items)}):**")
                        # Ограничиваем количество показываемых папок
                        max_folders = 10
                        displayed_items = items[:max_folders]
                        
                        for icon, name, path in displayed_items:
                            if st.button(f"{icon} {name}", key=f"dir_{path}", use_container_width=True):
                                st.session_state['current_folder'] = path
                                st.rerun()
                        
                        if len(items) > max_folders:
                            st.caption(f"... и еще {len(items) - max_folders} папок")
                    else:
                        st.info("В этой папке нет подпапок")
                
                # Кнопки управления
                col_select, col_cancel = st.columns(2)
                with col_select:
                    if st.button("✅ Выбрать эту папку", type="primary"):
                        st.session_state['show_browser'] = False
                        st.success(f"Выбрана папка: {current_dir}")
                        st.rerun()
                
                with col_cancel:
                    if st.button("❌ Отмена"):
                        st.session_state['show_browser'] = False
                        st.rerun()
                        
            except Exception as e:
                st.error(f"Ошибка при просмотре папки: {str(e)}")
        
        folder = st.session_state['current_folder']
        recursive = st.checkbox("Искать во вложенных папках", value=True)
        
        if st.button("Просканировать папку"):
            try:
                if not os.path.exists(folder):
                    st.error(f"Папка не найдена: {folder}")
                    st.session_state['paths'] = []
                elif not os.path.isdir(folder):
                    st.error(f"Указанный путь не является папкой: {folder}")
                    st.session_state['paths'] = []
                else:
                    # Рекурсивный или обычный поиск
                    if recursive:
                        pattern = os.path.join(folder, "**", "*.xlsx")
                        found_files = glob.glob(pattern, recursive=True)
                    else:
                        pattern = os.path.join(folder, "*.xlsx")
                        found_files = glob.glob(pattern)
                    
                    st.session_state['paths'] = sorted(found_files)
                    
                    if len(found_files) == 0:
                        st.warning(f"Excel файлы не найдены в папке: {folder}")
                    else:
                        st.success(f"Найдено файлов: {len(found_files)}")
                        
            except Exception as e:
                st.error(f"Ошибка при сканировании папки: {str(e)}")
                st.session_state['paths'] = []
        
        # Загружаем файлы, если они найдены
        paths = st.session_state.get('paths', [])
        if paths:
            # Показываем список найденных файлов
            with st.expander(f"📁 Найденные файлы ({len(paths)})", expanded=False):
                for p in paths:
                    st.text(f"📄 {os.path.relpath(p, folder)}")
            
            # Загружаем файлы
            for p in paths:
                files_data.append(load_employee_file_from_path(p))
            st.caption(f"✅ Загружено файлов: {len(paths)}")
        elif 'paths' in st.session_state and len(st.session_state['paths']) == 0:
            st.caption("🔍 Файлы не найдены")
    else:
        uploads = st.file_uploader("Загрузите Excel-отчёты (*.xlsx)", type=["xlsx"], accept_multiple_files=True)
        if uploads:
            for up in uploads:
                files_data.append(load_employee_file_from_bytes(up.name, up.read()))

with col2:
    ndfl_rate = st.number_input("Ставка НДФЛ (0.13 = 13%)", min_value=0.0, max_value=0.99, value=0.13, step=0.01)
    period = st.text_input("Период (подпись)", value=datetime.now().strftime("%Y-%m %d–%d"))
    st.caption("Онлайн-сводка: таблица обновляется при загрузке файлов. Gross-up = base/(1-rate).")

def update_totals_in_table(df, rate):
    """Обновляет итоговые строки в таблице сотрудников"""
    if len(df) == 0:
        return df
    
    # Убираем старые итоговые строки
    df_clean = df[~df["Сотрудник"].str.startswith("📊")].copy()
    
    # Если есть данные, добавляем новую итоговую строку
    if len(df_clean) > 0:
        total_row = pd.DataFrame([{
            "Сотрудник": "📊 ИТОГО",
            "Сумма из отчёта": df_clean["Сумма из отчёта"].sum().round(2),
            "Сумма с НДФЛ": df_clean["Сумма с НДФЛ"].sum().round(2)
        }])
        df_result = pd.concat([df_clean, total_row], ignore_index=True)
    else:
        df_result = df_clean
    
    return df_result

def calculate_tariff_totals(df):
    """Пересчитывает hero_total и importance_total как произведения тарифов на веса"""
    df = df.copy()
    
    # Убеждаемся, что колонки числовые
    numeric_cols = ['weight', 'importance', 'weight_tariff', 'importance_tariff', 'hero_total', 'resizes_total']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    # Пересчитываем колонки
    if all(col in df.columns for col in ['weight_tariff', 'weight']):
        df['hero_total'] = (df['weight_tariff'] * df['weight']).round(2)
    
    if all(col in df.columns for col in ['importance_tariff', 'importance']):
        df['importance_total'] = (df['importance_tariff'] * df['importance']).round(2)
    
    return df

def find_row_discrepancies(all_tasks, tolerance=0.01):
    """Находит строки с расхождениями между calculated_total и original_total"""
    discrepancies = []
    
    for idx, row in all_tasks.iterrows():
        # Пересчитываем для каждой строки
        hero_total = pd.to_numeric(row.get('hero_total', 0), errors='coerce') or 0
        resizes_total = pd.to_numeric(row.get('resizes_total', 0), errors='coerce') or 0
        original_total = pd.to_numeric(row.get('total', 0), errors='coerce') or 0
        
        calculated_total = hero_total + resizes_total
        difference = abs(calculated_total - original_total)
        
        if difference > tolerance:
            discrepancy = {
                'row_number': idx + 1,
                'ticket': str(row.get('ticket', 'N/A'))[:20] + ('...' if len(str(row.get('ticket', ''))) > 20 else ''),
                'assignee': str(row.get('assignee_name', row.get('assignee', 'N/A')))[:25] + ('...' if len(str(row.get('assignee_name', row.get('assignee', '')))) > 25 else ''),
                'summary': str(row.get('summary', 'N/A'))[:30] + ('...' if len(str(row.get('summary', ''))) > 30 else ''),
                'calculated_total': round(calculated_total, 2),
                'original_total': round(original_total, 2),
                'difference': round(difference, 2),
                'hero_total': round(hero_total, 2),
                'resizes_total': round(resizes_total, 2),
                'source_file': str(row.get('source_file', 'N/A'))
            }
            discrepancies.append(discrepancy)
    
    return discrepancies

def validate_final_totals(all_tasks_data, accounting_table, tolerance=0.01):
    """
    Финальная сверка сумм между All tasks и For accounting
    
    Args:
        all_tasks_data: список DataFrame с данными всех сотрудников
        accounting_table: DataFrame с итоговой таблицей для бухгалтерии
        tolerance: допустимое отклонение
        
    Returns:
        {
            'is_valid': bool,
            'all_tasks_total': float,
            'accounting_total': float,
            'difference': float,
            'employee_discrepancies': [{'employee': str, 'all_tasks': float, 'accounting': float, 'diff': float}]
        }
    """
    if not all_tasks_data or len(accounting_table) == 0:
        return {
            'is_valid': False,
            'error': 'Недостаточно данных для сверки',
            'all_tasks_total': 0.0,
            'accounting_total': 0.0,
            'difference': 0.0,
            'employee_discrepancies': []
        }
    
    # Объединяем все данные и очищаем
    all_tasks = pd.concat(all_tasks_data, ignore_index=True)
    
    # Убираем строки-итоги
    if 'ticket' in all_tasks.columns:
        mask = ~all_tasks['ticket'].astype(str).str.lower().str.contains(
            'итого|общая сумма|total|сумма:', na=False
        )
        all_tasks = all_tasks[mask]
    
    # Убираем пустые строки
    if 'assignee_name' in all_tasks.columns:
        mask = (all_tasks['assignee_name'].notna()) & (all_tasks['assignee_name'].str.strip() != '')
        all_tasks = all_tasks[mask]
    
    # Считаем итог из All tasks (исправляем данные на лету)
    corrected_tasks = all_tasks.copy()
    for idx, row in all_tasks.iterrows():
        ticket_validation = validate_ticket_unified(row, tolerance)
        if not ticket_validation['is_valid']:
            for correction in ticket_validation['corrections']:
                corrected_tasks.loc[idx, correction['field']] = correction['new_value']
    
    all_tasks_total = pd.to_numeric(corrected_tasks['total'], errors='coerce').fillna(0).sum()
    
    # Считаем итог из For accounting (убираем итоговые строки)
    accounting_clean = accounting_table[~accounting_table["Сотрудник"].str.startswith("📊")].copy()
    accounting_total = accounting_clean["Сумма из отчёта"].sum()
    
    # Общая разница
    difference = round(abs(all_tasks_total - accounting_total), 2)
    
    # Проверяем расхождения по сотрудникам
    employee_discrepancies = []
    for _, emp_row in accounting_clean.iterrows():
        emp_name = emp_row["Сотрудник"]
        emp_accounting_total = emp_row["Сумма из отчёта"]
        
        # Найдем задачи этого сотрудника в corrected_tasks
        if 'assignee_name' in corrected_tasks.columns:
            emp_tasks = corrected_tasks[corrected_tasks['assignee_name'].str.contains(emp_name, case=False, na=False)]
            emp_all_tasks_total = pd.to_numeric(emp_tasks['total'], errors='coerce').fillna(0).sum()
            
            emp_diff = round(abs(emp_accounting_total - emp_all_tasks_total), 2)
            if emp_diff > tolerance:
                employee_discrepancies.append({
                    'employee': emp_name,
                    'all_tasks': round(emp_all_tasks_total, 2),
                    'accounting': emp_accounting_total,
                    'diff': emp_diff
                })
    
    return {
        'is_valid': difference <= tolerance,
        'all_tasks_total': round(all_tasks_total, 2),
        'accounting_total': round(accounting_total, 2),
        'difference': difference,
        'employee_discrepancies': employee_discrepancies,
        'corrected_data': corrected_tasks
    }


def detailed_row_validation(row, tolerance=0.01):
    """
    Детальная валидация строки с типизацией ошибок
    
    Проверяет:
    1. weight × weight_tariff vs hero_total
    2. importance × importance_tariff vs resizes_total  
    3. (weight_calc + importance_calc) vs total
    
    Returns: список словарей с типизированными ошибками
    """
    errors = []
    
    # Извлекаем и конвертируем значения
    weight = pd.to_numeric(row.get('weight', 0), errors='coerce') or 0
    importance = pd.to_numeric(row.get('importance', 0), errors='coerce') or 0
    weight_tariff = pd.to_numeric(row.get('weight_tariff', 0), errors='coerce') or 0
    importance_tariff = pd.to_numeric(row.get('importance_tariff', 0), errors='coerce') or 0
    hero_total = pd.to_numeric(row.get('hero_total', 0), errors='coerce') or 0
    resizes_total = pd.to_numeric(row.get('resizes_total', 0), errors='coerce') or 0
    original_total = pd.to_numeric(row.get('total', 0), errors='coerce') or 0
    
    # Вычисляем ожидаемые значения
    weight_calculated = round(weight * weight_tariff, 2)
    importance_calculated = round(importance * importance_tariff, 2)
    total_calculated = round(weight_calculated + importance_calculated, 2)
    
    # Проверка 1: weight × weight_tariff vs hero_total
    weight_diff = abs(weight_calculated - hero_total)
    if weight_diff > tolerance:
        errors.append({
            'type': 'weight_mismatch',
            'field': 'hero_total',
            'description': f'weight({weight}) × weight_tariff({weight_tariff}) = {weight_calculated} ≠ hero_total({hero_total})',
            'expected': weight_calculated,
            'actual': hero_total,
            'difference': round(weight_diff, 2),
            'suggestion': f'Изменить hero_total с {hero_total} на {weight_calculated}',
            'severity': 'high' if weight_diff > 10 else 'medium' if weight_diff > 1 else 'low'
        })
    
    # Проверка 2: importance × importance_tariff vs resizes_total
    importance_diff = abs(importance_calculated - resizes_total)
    if importance_diff > tolerance:
        errors.append({
            'type': 'importance_mismatch',
            'field': 'resizes_total',
            'description': f'importance({importance}) × importance_tariff({importance_tariff}) = {importance_calculated} ≠ resizes_total({resizes_total})',
            'expected': importance_calculated,
            'actual': resizes_total,
            'difference': round(importance_diff, 2),
            'suggestion': f'Изменить resizes_total с {resizes_total} на {importance_calculated}',
            'severity': 'high' if importance_diff > 10 else 'medium' if importance_diff > 1 else 'low'
        })
    
    # Проверка 3: Общая сумма
    total_diff = abs(total_calculated - original_total)
    if total_diff > tolerance:
        errors.append({
            'type': 'total_mismatch',
            'field': 'total',
            'description': f'Сумма компонентов {total_calculated} ≠ total({original_total})',
            'expected': total_calculated,
            'actual': original_total,
            'difference': round(total_diff, 2),
            'suggestion': f'Изменить total с {original_total} на {total_calculated}',
            'severity': 'high' if total_diff > 50 else 'medium' if total_diff > 5 else 'low'
        })
    
    return errors

def analyze_detailed_validation_results(all_tasks, tolerance=0.01):
    """
    Анализирует все строки и группирует ошибки по типам
    
    Returns: детальный отчет с группировкой и статистикой
    """
    all_errors = []
    error_stats = {
        'weight_mismatch': {'count': 0, 'total_diff': 0.0, 'errors': []},
        'importance_mismatch': {'count': 0, 'total_diff': 0.0, 'errors': []},
        'total_mismatch': {'count': 0, 'total_diff': 0.0, 'errors': []},
    }
    severity_stats = {'high': 0, 'medium': 0, 'low': 0}
    
    # Анализируем каждую строку
    for idx, row in all_tasks.iterrows():
        row_errors = detailed_row_validation(row, tolerance)
        
        if row_errors:
            # Добавляем контекстную информацию к ошибкам
            for error in row_errors:
                error['row_number'] = idx + 1
                error['ticket'] = str(row.get('ticket', 'N/A'))[:25] + ('...' if len(str(row.get('ticket', ''))) > 25 else '')
                error['assignee'] = str(row.get('assignee_name', row.get('assignee', 'N/A')))[:25] + ('...' if len(str(row.get('assignee_name', row.get('assignee', '')))) > 25 else '')
                error['summary'] = str(row.get('summary', 'N/A'))[:35] + ('...' if len(str(row.get('summary', ''))) > 35 else '')
                error['source_file'] = str(row.get('source_file', 'N/A'))
                
                # Группируем по типам
                error_type = error['type']
                if error_type in error_stats:
                    error_stats[error_type]['count'] += 1
                    error_stats[error_type]['total_diff'] += error['difference']
                    error_stats[error_type]['errors'].append(error)
                
                # Считаем по уровням критичности
                severity_stats[error['severity']] += 1
                all_errors.append(error)
    
    # Формируем итоговый отчет
    result = {
        'total_rows_checked': len(all_tasks),
        'total_errors_found': len(all_errors),
        'rows_with_errors': len(set(error['row_number'] for error in all_errors)),
        'error_rate': round(len(set(error['row_number'] for error in all_errors)) / len(all_tasks) * 100, 1) if len(all_tasks) > 0 else 0,
        'all_errors': all_errors,
        'error_stats': {
            error_type: {
                'count': stats['count'],
                'avg_difference': round(stats['total_diff'] / stats['count'], 2) if stats['count'] > 0 else 0,
                'total_difference': round(stats['total_diff'], 2),
                'errors': stats['errors']
            }
            for error_type, stats in error_stats.items()
        },
        'severity_stats': severity_stats,
        'summary': {
            'most_common_error': max(error_stats.keys(), key=lambda x: error_stats[x]['count']) if any(error_stats[x]['count'] > 0 for x in error_stats) else None,
            'total_financial_impact': round(sum(error_stats[x]['total_diff'] for x in error_stats), 2),
            'files_affected': len(set(error['source_file'] for error in all_errors))
        }
    }
    
    return result

def apply_auto_corrections(all_tasks, error_analysis, correction_types=['weight_mismatch', 'importance_mismatch', 'total_mismatch']):
    """
    Применяет автоматические исправления к данным
    
    Args:
        all_tasks: DataFrame с данными
        error_analysis: результат analyze_detailed_validation_results
        correction_types: список типов ошибок для исправления
    
    Returns: 
        {
            'corrected_data': DataFrame с исправленными данными,
            'corrections_applied': список примененных исправлений,
            'corrections_summary': сводка по исправлениям
        }
    """
    corrected_data = all_tasks.copy()
    corrections_applied = []
    corrections_summary = {
        'total_corrections': 0,
        'by_type': {error_type: 0 for error_type in correction_types},
        'financial_impact': 0.0
    }
    
    # Применяем исправления для каждого типа ошибок
    for error_type in correction_types:
        if error_type in error_analysis['error_stats']:
            errors = error_analysis['error_stats'][error_type]['errors']
            
            for error in errors:
                row_idx = error['row_number'] - 1  # Конвертируем в 0-based индекс
                
                if row_idx < len(corrected_data):
                    # Запоминаем исходное значение
                    old_value = corrected_data.iloc[row_idx][error['field']]
                    
                    # Применяем исправление
                    corrected_data.iloc[row_idx, corrected_data.columns.get_loc(error['field'])] = error['expected']
                    
                    # Записываем информацию об исправлении
                    correction_record = {
                        'row_number': error['row_number'],
                        'ticket': error['ticket'],
                        'assignee': error['assignee'],
                        'field': error['field'],
                        'error_type': error_type,
                        'old_value': old_value,
                        'new_value': error['expected'],
                        'difference': error['difference'],
                        'description': error['description'],
                        'source_file': error['source_file']
                    }
                    corrections_applied.append(correction_record)
                    
                    # Обновляем статистику
                    corrections_summary['total_corrections'] += 1
                    corrections_summary['by_type'][error_type] += 1
                    corrections_summary['financial_impact'] += abs(error['difference'])
    
    # Округляем финансовый эффект
    corrections_summary['financial_impact'] = round(corrections_summary['financial_impact'], 2)
    
    return {
        'corrected_data': corrected_data,
        'corrections_applied': corrections_applied,
        'corrections_summary': corrections_summary
    }

def generate_corrections_report(corrections_applied):
    """
    Генерирует детальный отчет о примененных исправлениях
    """
    if not corrections_applied:
        return "Исправления не применялись."
    
    report_lines = []
    report_lines.append("📋 ОТЧЕТ О ПРИМЕНЕННЫХ ИСПРАВЛЕНИЯХ")
    report_lines.append("=" * 50)
    
    # Группируем по файлам
    by_file = {}
    for correction in corrections_applied:
        file = correction['source_file']
        if file not in by_file:
            by_file[file] = []
        by_file[file].append(correction)
    
    # Отчет по файлам
    for file, file_corrections in by_file.items():
        report_lines.append(f"\n📁 Файл: {file} ({len(file_corrections)} исправлений)")
        report_lines.append("-" * 40)
        
        for correction in file_corrections:
            report_lines.append(f"  • Строка {correction['row_number']}: {correction['assignee']} | {correction['ticket']}")
            report_lines.append(f"    Поле: {correction['field']}")
            report_lines.append(f"    Было: {correction['old_value']} → Стало: {correction['new_value']}")
            report_lines.append(f"    Разница: {round(abs(correction['new_value'] - correction['old_value']), 2)}")
            report_lines.append(f"    Причина: {correction['description']}")
            report_lines.append("")
    
    # Общая статистика
    total_financial_impact = sum(abs(c['new_value'] - c['old_value']) for c in corrections_applied)
    report_lines.append(f"\n💰 ИТОГО:")
    report_lines.append(f"  • Всего исправлений: {len(corrections_applied)}")
    report_lines.append(f"  • Финансовый эффект: {total_financial_impact:.2f} ₽")
    report_lines.append(f"  • Файлов затронуто: {len(by_file)}")
    
    # Статистика по типам
    by_type = {}
    for correction in corrections_applied:
        error_type = correction['error_type']
        if error_type not in by_type:
            by_type[error_type] = 0
        by_type[error_type] += 1
    
    report_lines.append(f"\n📊 По типам ошибок:")
    type_names = {
        'weight_mismatch': 'Ошибки в hero_total',
        'importance_mismatch': 'Ошибки в resizes_total', 
        'total_mismatch': 'Ошибки в общей сумме'
    }
    
    for error_type, count in by_type.items():
        type_name = type_names.get(error_type, error_type)
        report_lines.append(f"  • {type_name}: {count}")
    
    return "\n".join(report_lines)


def compare_and_correct_export(personal_df, export_df, tolerance=0.01):
    """
    Сравнивает личный отчет и отчет из выгрузки по тикету и корректирует ТОЛЬКО выгрузку
    на основе личной таблицы, применяя формулы:
      - weight × weight_tariff = hero_total
      - importance × importance_tariff = resizes_total
      - hero_total + resizes_total = total

    Args:
        personal_df: DataFrame личного отчета (из файла сотрудника)
        export_df: DataFrame отчета из выгрузки
        tolerance: допустимое отклонение для сравнения total

    Returns:
        {
          'corrected_export': DataFrame,                  # скорректированная выгрузка
          'comparison_report': DataFrame,                 # построчные изменения (по полям)
          'only_in_personal': DataFrame,                  # тикеты без пары в выгрузке
          'only_in_export': DataFrame,                    # тикеты без пары в личной
          'summary': { 'matched': int, 'changed_rows': int, 'total_changes': int }
        }
    """
    # Подготовка данных: убираем строки-итоги, канонизируем колонки
    def _prep(df):
        df = df.copy()
        df = canonize_columns(df)
        if 'ticket' in df.columns:
            mask = ~df['ticket'].astype(str).str.lower().str.contains('итого|общая сумма|total|сумма:', na=False)
            df = df[mask]
        # Гарантируем наличие необходимых колонок
        for col in DEF_COLS:
            if col not in df.columns:
                df[col] = 0.0 if col in ['weight','importance','weight_tariff','importance_tariff','hero_total','resizes_total','total','difficulty'] else ""
        # Числовые столбцы -> float
        for col in ['weight','importance','weight_tariff','importance_tariff','hero_total','resizes_total','total','difficulty']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        # Тикеты как строки
        if 'ticket' in df.columns:
            df['ticket'] = df['ticket'].astype(str).str.strip()
        else:
            df['ticket'] = ""
        return df

    p = _prep(personal_df)
    e = _prep(export_df)

    # Совпадения по тикету
    p_keys = set(p['ticket'])
    e_keys = set(e['ticket'])
    matched_keys = sorted(list(p_keys & e_keys))
    only_in_p = p[~p['ticket'].isin(e_keys)].copy()
    only_in_e = e[~e['ticket'].isin(p_keys)].copy()

    corrected_export = e.copy()
    changes = []  # список записей об изменениях по полям

    compare_fields = ['difficulty','weight','importance','hero_total','resizes_total','total']

    for key in matched_keys:
        row_p = p[p['ticket'] == key].iloc[0]
        idx_e = corrected_export.index[corrected_export['ticket'] == key][0]
        row_e = corrected_export.loc[idx_e]

        # Фиксируем различия по заданным полям
        for fld in compare_fields:
            val_p = pd.to_numeric(row_p.get(fld, 0), errors='coerce') if fld != 'ticket' else str(row_p.get('ticket',''))
            val_e = pd.to_numeric(row_e.get(fld, 0), errors='coerce') if fld != 'ticket' else str(row_e.get('ticket',''))
            if (isinstance(val_p, (int,float)) and isinstance(val_e, (int,float)) and abs(float(val_p) - float(val_e)) > tolerance) or (not isinstance(val_p, (int,float)) and val_p != val_e):
                changes.append({
                    'ticket': key,
                    'field': fld,
                    'export_value': float(val_e) if isinstance(val_e, (int,float)) else val_e,
                    'personal_value': float(val_p) if isinstance(val_p, (int,float)) else val_p,
                    'change_applied': False,
                    'reason': 'Поле отличается'
                })

        # Если расходится total — пересчитываем по данным личной таблицы
        total_p = pd.to_numeric(row_p.get('total', 0), errors='coerce') or 0.0
        total_e = pd.to_numeric(row_e.get('total', 0), errors='coerce') or 0.0

        if abs(total_p - total_e) > tolerance:
            w = pd.to_numeric(row_p.get('weight', 0), errors='coerce') or 0.0
            imp = pd.to_numeric(row_p.get('importance', 0), errors='coerce') or 0.0
            wt = pd.to_numeric(row_p.get('weight_tariff', 0), errors='coerce') or 0.0
            it = pd.to_numeric(row_p.get('importance_tariff', 0), errors='coerce') or 0.0

            hero_calc = round(w * wt, 2)
            resizes_calc = round(imp * it, 2)
            total_calc = round(hero_calc + resizes_calc, 2)

            # Переносим базовые поля из личной для консистентности
            for fld, new_val in [
                ('weight', w),
                ('importance', imp),
                ('weight_tariff', wt),
                ('importance_tariff', it),
                ('hero_total', hero_calc),
                ('resizes_total', resizes_calc),
                ('total', total_calc),
            ]:
                old_val = pd.to_numeric(corrected_export.loc[idx_e, fld], errors='coerce') if fld != 'ticket' else corrected_export.loc[idx_e, fld]
                if (isinstance(old_val, (int,float)) and abs(float(old_val) - float(new_val)) > tolerance) or (not isinstance(old_val, (int,float)) and old_val != new_val):
                    corrected_export.loc[idx_e, fld] = new_val
                    changes.append({
                        'ticket': key,
                        'field': fld,
                        'export_value': float(old_val) if isinstance(old_val, (int,float)) else old_val,
                        'personal_value': float(row_p.get(fld, new_val)) if isinstance(row_p.get(fld, new_val), (int,float)) else row_p.get(fld, new_val),
                        'new_export_value': float(new_val) if isinstance(new_val, (int,float)) else new_val,
                        'change_applied': True,
                        'reason': 'Коррекция по формуле из личного отчета'
                    })

    comparison_report = pd.DataFrame(changes)
    summary = {
        'matched': len(matched_keys),
        'changed_rows': len(set(comparison_report[comparison_report['change_applied'] == True]['ticket'])) if not comparison_report.empty else 0,
        'total_changes': len(comparison_report[comparison_report['change_applied'] == True]) if not comparison_report.empty else 0
    }

    return {
        'corrected_export': corrected_export,
        'comparison_report': comparison_report,
        'only_in_personal': only_in_p[DEF_COLS + ['ticket']].copy() if len(only_in_p) else only_in_p,
        'only_in_export': only_in_e[DEF_COLS + ['ticket']].copy() if len(only_in_e) else only_in_e,
        'summary': summary
    }

def validate_sums(files_data, accounting_table):
    """Тройная валидация сумм: пересчет тарифов vs total vs accounting"""
    if not files_data or len(accounting_table) == 0:
        return {"status": "no_data", "message": "Недостаточно данных для сверки"}
    
    # Подсчитываем сумму из All tasks
    all_tasks = pd.concat(files_data, ignore_index=True)
    
    # Убираем строки-итоги
    if 'ticket' in all_tasks.columns:
        mask = ~all_tasks['ticket'].astype(str).str.lower().str.contains(
            'итого|общая сумма|total|сумма:', na=False
        )
        all_tasks = all_tasks[mask]
    
    # Убираем пустые строки
    if 'assignee_name' in all_tasks.columns:
        mask = (all_tasks['assignee_name'].notna()) & (all_tasks['assignee_name'].str.strip() != '')
        all_tasks = all_tasks[mask]
    
    # Пересчитываем hero_total и importance_total
    all_tasks = calculate_tariff_totals(all_tasks)
    
    # Считаем все три суммы
    all_tasks_total = pd.to_numeric(all_tasks['total'], errors='coerce').fillna(0).sum() if 'total' in all_tasks.columns else 0.0
    
    # Принудительно конвертируем в числовые значения перед суммированием
    if all(col in all_tasks.columns for col in ['hero_total', 'resizes_total']):
        hero_sum = pd.to_numeric(all_tasks['hero_total'], errors='coerce').fillna(0).sum()
        resizes_sum = pd.to_numeric(all_tasks['resizes_total'], errors='coerce').fillna(0).sum()
        calculated_total = hero_sum + resizes_sum
    else:
        calculated_total = 0.0
    
    # Подсчитываем сумму из For accounting (убираем итоговые строки)
    accounting_clean = accounting_table[~accounting_table["Сотрудник"].str.startswith("📊")].copy()
    accounting_total = accounting_clean["Сумма из отчёта"].sum()
    
    # Тройное сравнение
    tolerance = 0.01  # Допустимая погрешность в 1 копейку
    
    diff_calc_vs_total = abs(calculated_total - all_tasks_total)
    diff_total_vs_accounting = abs(all_tasks_total - accounting_total)
    diff_calc_vs_accounting = abs(calculated_total - accounting_total)
    
    result = {
        "calculated_total": round(calculated_total, 2),
        "all_tasks_total": round(all_tasks_total, 2),
        "accounting_total": round(accounting_total, 2),
        "diff_calc_vs_total": round(diff_calc_vs_total, 2),
        "diff_total_vs_accounting": round(diff_total_vs_accounting, 2),
        "diff_calc_vs_accounting": round(diff_calc_vs_accounting, 2),
        "is_fully_valid": all(d <= tolerance for d in [diff_calc_vs_total, diff_total_vs_accounting, diff_calc_vs_accounting])
    }
    
    # Определяем статус и сообщение
    issues = []
    if diff_calc_vs_total > tolerance:
        issues.append(f"пересчет ≠ total ({diff_calc_vs_total:.2f}₽)")
    if diff_total_vs_accounting > tolerance:
        issues.append(f"total ≠ accounting ({diff_total_vs_accounting:.2f}₽)")
    if diff_calc_vs_accounting > tolerance:
        issues.append(f"пересчет ≠ accounting ({diff_calc_vs_accounting:.2f}₽)")
    
    if result["is_fully_valid"]:
        result["status"] = "success"
        result["message"] = "✅ Все суммы сходятся! Тройная проверка пройдена."
    else:
        result["status"] = "error"
        result["message"] = f"❌ Обнаружены расхождения: {', '.join(issues)}"
        
        # Детализация расхождений
        details = []
        details.append("🔢 Сравнение сумм:")
        details.append(f"• Пересчет тарифов: {calculated_total:.2f} ₽")
        details.append(f"• Total из файлов: {all_tasks_total:.2f} ₽")
        details.append(f"• For accounting: {accounting_total:.2f} ₽")
        details.append("")
        
        # Проверим сходимость по сотрудникам только если есть базовые расхождения
        if diff_total_vs_accounting > tolerance:
            details.append("👤 Расхождения по сотрудникам (total vs accounting):")
            employee_issues = []
            for _, emp_row in accounting_clean.iterrows():
                emp_name = emp_row["Сотрудник"]
                emp_total_accounting = emp_row["Сумма из отчёта"]
                
                # Найдем задачи этого сотрудника в all_tasks
                if 'assignee_name' in all_tasks.columns:
                    emp_tasks = all_tasks[all_tasks['assignee_name'].str.contains(emp_name, case=False, na=False)]
                    emp_total_tasks = emp_tasks['total'].sum()
                    
                    emp_diff = abs(emp_total_accounting - emp_total_tasks)
                    if emp_diff > tolerance:
                        employee_issues.append({
                            "employee": emp_name,
                            "accounting": emp_total_accounting,
                            "tasks": round(emp_total_tasks, 2),
                            "diff": round(emp_diff, 2)
                        })
            
            if employee_issues:
                for issue in employee_issues[:5]:  # Показываем первых 5
                    details.append(f"• {issue['employee']}: отчёт {issue['accounting']} ≠ задачи {issue['tasks']} ({issue['diff']}₽)")
            else:
                details.append("• Расхождения по сотрудникам не найдены")
        
        # Проверим тарифные расхождения
        if diff_calc_vs_total > tolerance:
            details.append("")
            details.append("⚙️ Проблемы с тарифными расчетами:")
            details.append("• Сумма hero_total + resizes_total не равна total")
            
            # Построчный анализ проблемных задач
            row_discrepancies = find_row_discrepancies(all_tasks, tolerance)
            if row_discrepancies:
                details.append("")
                details.append("🔍 Проблемные строки (первые 10):")
                for i, disc in enumerate(row_discrepancies[:10]):
                    details.append(f"• Строка {disc['row_number']}: {disc['assignee']} | {disc['ticket']} | {disc['summary']}")
                    details.append(f"  📊 Пересчет: {disc['calculated_total']}₽ vs Оригинал: {disc['original_total']}₽ (разница {disc['difference']}₽)")
                    details.append(f"  📋 Детали: hero={disc['hero_total']}₽ + resizes={disc['resizes_total']}₽")
                    details.append(f"  📁 Файл: {disc['source_file']}")
                    if i < len(row_discrepancies[:10]) - 1:
                        details.append("")
                
                if len(row_discrepancies) > 10:
                    details.append(f"... и еще {len(row_discrepancies) - 10} проблемных строк")
            else:
                details.append("• Проблемные строки не найдены (проверьте общую логику расчета)")
        
        result["details"] = details
        result["row_discrepancies"] = find_row_discrepancies(all_tasks, tolerance) if diff_calc_vs_total > tolerance else []
    
    return result

def build_summary(files_data, rate):
    # Создаем пустую таблицу, если нет файлов
    if not files_data:
        return pd.DataFrame(columns=["Сотрудник", "Сумма из отчёта", "Сумма с НДФЛ"])
    
    # Объединяем все задачи из файлов
    all_tasks = pd.concat(files_data, ignore_index=True)
    if 'assignee_name' not in all_tasks:
        all_tasks['assignee_name'] = ""
    if 'total' not in all_tasks:
        all_tasks['total'] = 0.0
    
    # Очищаем имена и группируем по сотрудникам
    all_tasks['assignee_name'] = all_tasks['assignee_name'].apply(clean_name)
    by = all_tasks.groupby('assignee_name', dropna=False)['total'].sum().reset_index()
    by.rename(columns={'assignee_name':'Сотрудник','total':'Сумма из отчёта'}, inplace=True)
    
    # Убираем пустые имена, технические значения и сортируем алфавитно
    tech_values = ['', 'string', 'nan', 'none', 'null', 'undefined']
    mask = by['Сотрудник'].str.strip().str.lower().isin(tech_values) == False
    mask = mask & (by['Сотрудник'].str.strip() != '')
    by = by[mask].copy()
    by = by.sort_values('Сотрудник').reset_index(drop=True)
    
    # Рассчитываем gross-up НДФЛ
    by["Сумма с НДФЛ"] = (by["Сумма из отчёта"] / (1 - rate)).round(2)
    by["Сумма из отчёта"] = by["Сумма из отчёта"].round(2)
    
    # Добавляем итоговую строку, если есть данные
    if len(by) > 0:
        total_row = pd.DataFrame([{
            "Сотрудник": "📊 ИТОГО",
            "Сумма из отчёта": by["Сумма из отчёта"].sum().round(2),
            "Сумма с НДФЛ": by["Сумма с НДФЛ"].sum().round(2)
        }])
        by = pd.concat([by, total_row], ignore_index=True)
    
    return by

summary_df = build_summary(files_data, ndfl_rate)

if "custom_table" not in st.session_state:
    st.session_state["custom_table"] = summary_df.copy()
else:
    old = st.session_state["custom_table"]
    
    # Сохраняем текущий порядок сотрудников из пользовательской таблицы
    existing_names = list(old["Сотрудник"]) if len(old) > 0 else []
    
    # Если есть новые данные из файлов, обновляем суммы
    if len(summary_df) > 0:
        # Объединяем с новыми данными, сохраняя пользовательские изменения имен
        merged = old.merge(summary_df, on="Сотрудник", how="outer", suffixes=("","_new"))
        
        # Обновляем суммы из файлов
        for col in ["Сумма из отчёта","Сумма с НДФЛ"]:
            merged[col] = merged.get(f"{col}_new", pd.Series(dtype=float)).fillna(merged[col]).fillna(0.0)
            if f"{col}_new" in merged.columns:
                merged.drop(columns=[f"{col}_new"], inplace=True)
        
        # Сохраняем порядок: сначала существующие, потом новые
        new_names = [n for n in list(summary_df["Сотрудник"]) if n not in existing_names]
        ordered_names = existing_names + new_names
        merged["__order"] = merged["Сотрудник"].apply(lambda x: ordered_names.index(x) if x in ordered_names else 10**6)
        merged = merged.sort_values("__order").drop(columns=["__order"]).reset_index(drop=True)
        
        st.session_state["custom_table"] = merged
    # Если новых данных нет, но таблица есть, оставляем как есть для пользовательского редактирования

st.header("Список сотрудников (онлайн)")
st.caption("Таблица обновляется в реальном времени при загрузке файлов. Изначально пуста, сотрудники появляются автоматически.")

edited = st.data_editor(
    st.session_state["custom_table"],
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Сотрудник": st.column_config.TextColumn("Сотрудник", required=False),
        "Сумма из отчёта": st.column_config.NumberColumn("Сумма из отчёта", disabled=True, format="%.2f"),
        "Сумма с НДФЛ": st.column_config.NumberColumn("Сумма с НДФЛ", disabled=True, format="%.2f"),
    },
    hide_index=False,
    key="editor_table_live",
)

st.session_state["custom_table"]["Сотрудник"] = edited["Сотрудник"]

# Обновляем итоговые суммы после любых изменений в таблице
st.session_state["custom_table"] = update_totals_in_table(st.session_state["custom_table"], ndfl_rate)

st.subheader("Управление порядком / составом")

# CSS для выравнивания кнопок и стилизации таблицы
st.markdown("""
<style>
    /* Выравниваем все элементы управления по базовой линии */
    div[data-testid="column"] {
        display: flex;
        align-items: flex-end;
    }
    
    /* Единая высота для всех кнопок */
    .stButton > button {
        height: 38px !important;
        margin-bottom: 0px !important;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    /* Выравниваем селектор */
    .stSelectbox > div > div {
        margin-bottom: 0px !important;
    }
    
    /* Выравниваем текстовое поле */
    .stTextInput > div > div {
        margin-bottom: 0px !important;
    }
    
    /* СКРЫВАЕМ автоматические лейблы Streamlit для пустых строк */
    label[data-testid="stWidgetLabel"]:empty {
        display: none !important;
    }
    
    /* Дополнительно скрываем лейблы с пустым текстом */
    label[data-testid="stWidgetLabel"]:has(> p:empty) {
        display: none !important;
    }
    
    /* Универсальное скрытие лейблов в блоке управления */
    div[data-testid="stHorizontalBlock"] label[data-testid="stWidgetLabel"] {
        display: none !important;
        height: 0px !important;
        margin: 0px !important;
        padding: 0px !important;
    }
    
    /* Стилизация итоговой строки в таблице */
    [data-testid="data-editor"] tbody tr:last-child {
        background-color: #f0f2f6 !important;
        border-top: 2px solid #1f77b4 !important;
        font-weight: bold !important;
    }
    
    /* Улучшенная стилизация строк таблицы */
    [data-testid="data-editor"] tbody tr:hover {
        background-color: #e8f4f8 !important;
    }
</style>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns([3,0.8,0.8,2])
# Исключаем итоговую строку из управления
all_names = list(st.session_state["custom_table"]["Сотрудник"])
names_for_select = [name for name in all_names if not name.startswith("📊")]
with c1:
    sel_name = st.selectbox("", names_for_select if names_for_select else ["—"], placeholder="Выбрать строку")
with c2:
    if st.button("⬆️", help="Вверх", use_container_width=True) and names_for_select and sel_name in names_for_select:
        df = st.session_state["custom_table"]
        idx = df.index[df["Сотрудник"]==sel_name][0]
        # Не позволяем перемещать выше первой строки или затрагивать итоговую строку
        if idx > 0 and not df.iloc[idx-1]["Сотрудник"].startswith("📊"):
            df.iloc[[idx-1, idx]] = df.iloc[[idx, idx-1]].values
            st.session_state["custom_table"] = update_totals_in_table(df.reset_index(drop=True), ndfl_rate)
with c3:
    if st.button("⬇️", help="Вниз", use_container_width=True) and names_for_select and sel_name in names_for_select:
        df = st.session_state["custom_table"]
        idx = df.index[df["Сотрудник"]==sel_name][0]
        # Не позволяем перемещать ниже последней обычной строки (до итоговой)
        total_rows = df[df["Сотрудник"].str.startswith("📊")]
        max_idx = len(df) - len(total_rows) - 1 if len(total_rows) > 0 else len(df) - 1
        if idx < max_idx:
            df.iloc[[idx, idx+1]] = df.iloc[[idx+1, idx]].values
            st.session_state["custom_table"] = update_totals_in_table(df.reset_index(drop=True), ndfl_rate)
with c4:
    if st.button("🗑️ Удалить", help="Удалить выбранную строку", use_container_width=True) and names_for_select and sel_name in names_for_select:
        df = st.session_state["custom_table"]
        st.session_state["custom_table"] = update_totals_in_table(df[df["Сотрудник"]!=sel_name].reset_index(drop=True), ndfl_rate)

st.markdown("— или —")
a1, a2 = st.columns([3,1])
with a1:
    new_name = st.text_input("", placeholder="ФИО нового сотрудника")
with a2:
    if st.button("➕ Добавить", help="Добавить новую строку", use_container_width=True):
        df = st.session_state["custom_table"]
        new_df = pd.concat([df, pd.DataFrame([{
            "Сотрудник": new_name.strip() if new_name else "",
            "Сумма из отчёта": 0.0,
            "Сумма с НДФЛ": 0.0,
        }])], ignore_index=True)
        st.session_state["custom_table"] = update_totals_in_table(new_df, ndfl_rate)

st.markdown("---")

# 🎯 УНИФИЦИРОВАННАЯ ВАЛИДАЦИЯ
st.markdown("### 🎯 Унифицированная система валидации")
st.caption("Трехуровневая проверка: тикеты → сотрудники → финальная сверка")

if files_data:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🔍 Запустить полную валидацию", help="Трехуровневая проверка всех данных"):
            # Запускаем финальную сверку с автоисправлениями
            validation_result = validate_final_totals(files_data, st.session_state["custom_table"], tolerance=0.01)
            
            if 'error' in validation_result:
                st.warning(f"⚠️ {validation_result['error']}")
            elif validation_result['is_valid']:
                st.success("✅ Все суммы сходятся! Валидация пройдена успешно.")
                
                col1, col2 = st.columns(2) 
                with col1:
                    st.metric("All tasks (исправлено)", f"{validation_result['all_tasks_total']:.2f} ₽")
                with col2:
                    st.metric("For accounting", f"{validation_result['accounting_total']:.2f} ₽")
                    
                # Сохраняем исправленные данные для экспорта
                st.session_state['unified_validation_result'] = validation_result
                
            else:
                st.error(f"❌ Обнаружены расхождения: {validation_result['difference']:.2f} ₽")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("All tasks (исправлено)", f"{validation_result['all_tasks_total']:.2f} ₽", delta=f"{validation_result['difference']:.2f} ₽")
                with col2:
                    st.metric("For accounting", f"{validation_result['accounting_total']:.2f} ₽")
                
                # Показываем расхождения по сотрудникам
                if validation_result['employee_discrepancies']:
                    with st.expander(f"👤 Расхождения по сотрудникам ({len(validation_result['employee_discrepancies'])})", expanded=True):
                        for disc in validation_result['employee_discrepancies']:
                            st.write(f"**{disc['employee']}**: All tasks = {disc['all_tasks']} ₽, Accounting = {disc['accounting']} ₽, разница = {disc['diff']} ₽")
                
                # Сохраняем результат
                st.session_state['unified_validation_result'] = validation_result
    
    with col2:
        st.caption("🔧 **Что проверяется:**\n• weight × tariff = hero_total\n• importance × tariff = resizes_total\n• hero + resizes = total\n• Сверка с For accounting")

# Старый блок удален - используется унифицированная валидация выше

# Старый блок детальной валидации удален - используется унифицированная система выше

st.markdown("---")

# Старый блок проверки отдельного отчета заменен на улучшенную систему ниже ⬇️
st.info("💡 Для проверки отдельных отчетов используйте блоки выше и ниже: \n• Автономный инструмент для загрузки нового файла \n• Массовое скачивание исправленных отчетов")

st.markdown("---")

# 📥 СКАЧИВАНИЕ ИСПРАВЛЕННЫХ ОТЧЕТОВ СОТРУДНИКОВ
st.markdown("### 📥 Индивидуальные исправленные отчеты")
st.caption("Скачайте исправленные версии отчетов для отправки конкретным сотрудникам")

if files_data:
    # Сначала запускаем валидацию для получения исправленных данных
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if st.button("🔄 Подготовить исправленные отчеты", help="Валидация и подготовка всех отчетов"):
            # Применяем валидацию ко всем файлам
            individual_reports = {}
            
            for i, file_data in enumerate(files_data):
                try:
                    # Получаем имя файла
                    source_file = file_data.get('source_file', ['unknown'])[0] if 'source_file' in file_data.columns else f'Файл {i+1}'
                    
                    # Валидируем отчет сотрудника
                    employee_validation = validate_employee_report(file_data, tolerance=0.01)
                    
                    individual_reports[source_file] = {
                        'validation': employee_validation,
                        'file_index': i,
                        'has_errors': not employee_validation['is_valid'],
                        'employee_name': employee_validation['employee_name']
                    }
                    
                except Exception as e:
                    st.error(f"Ошибка при обработке файла {i+1}: {str(e)}")
            
            # Сохраняем результаты
            st.session_state['individual_reports'] = individual_reports
            
            # Показываем статистику
            total_files = len(individual_reports)
            files_with_errors = sum(1 for r in individual_reports.values() if r['has_errors'])
            
            if files_with_errors == 0:
                st.success(f"✅ Все {total_files} отчетов корректны!")
            else:
                st.warning(f"⚠️ Из {total_files} отчетов найдены ошибки в {files_with_errors} файлах")
    
    with col2:
        st.caption("Процесс:\n• Валидация всех файлов\n• Автоисправление ошибок\n• Подготовка к скачиванию")
    
    # Показываем список отчетов для скачивания
    if 'individual_reports' in st.session_state:
        individual_reports = st.session_state['individual_reports']
        
        if individual_reports:
            st.markdown("#### 📋 Список отчетов для скачивания")
            
            # Группируем по статусу
            correct_reports = {k: v for k, v in individual_reports.items() if not v['has_errors']}
            corrected_reports = {k: v for k, v in individual_reports.items() if v['has_errors']}
            
            if correct_reports:
                with st.expander(f"✅ Корректные отчеты ({len(correct_reports)})", expanded=False):
                    for file_name, report_data in correct_reports.items():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"📄 **{file_name}** - {report_data['employee_name']}")
                        with col2:
                            # Кнопка скачивания оригинального файла
                            original_data = files_data[report_data['file_index']]
                            export_data = original_data[DEF_COLS].copy()
                            
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                export_data.to_excel(writer, sheet_name="Report", index=False)
                            
                            st.download_button(
                                label="📥",
                                data=output.getvalue(),
                                file_name=f"verified_{file_name}",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                help=f"Скачать проверенный отчет {report_data['employee_name']}",
                                key=f"download_verified_{report_data['file_index']}"
                            )
            
            if corrected_reports:
                with st.expander(f"🔧 Исправленные отчеты ({len(corrected_reports)})", expanded=True):
                    for file_name, report_data in corrected_reports.items():
                        validation_result = report_data['validation']
                        
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.write(f"📄 **{file_name}** - {report_data['employee_name']}")
                            st.caption(f"Ошибок в тикетах: {len(validation_result['task_errors'])}, разница: {validation_result['difference']:.2f} ₽")
                        
                        with col2:
                            # Кнопка просмотра ошибок
                            if st.button("🔍", help="Показать детали ошибок", key=f"show_errors_{report_data['file_index']}"):
                                st.session_state[f'show_details_{report_data["file_index"]}'] = True
                        
                        with col3:
                            # Кнопка скачивания исправленного файла
                            corrected_data = validation_result['corrected_data']
                            export_data = corrected_data[DEF_COLS].copy()
                            
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                export_data.to_excel(writer, sheet_name="Corrected Report", index=False)
                            
                            st.download_button(
                                label="📥 Исправ.",
                                data=output.getvalue(),
                                file_name=f"corrected_{file_name}",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                help=f"Скачать исправленный отчет {report_data['employee_name']}",
                                key=f"download_corrected_{report_data['file_index']}"
                            )
                        
                        # Показываем детали ошибок, если запрошено
                        if st.session_state.get(f'show_details_{report_data["file_index"]}', False):
                            with st.container():
                                st.write("**Детали ошибок:**")
                                for error in validation_result['task_errors'][:5]:
                                    st.write(f"• Строка {error['row_idx']}: {error['ticket']}")
                                    for correction in error['corrections']:
                                        st.write(f"  - {correction['field']}: {correction['old_value']} → {correction['new_value']}")
                                if len(validation_result['task_errors']) > 5:
                                    st.caption(f"... и еще {len(validation_result['task_errors']) - 5} ошибок")
                                
                                if st.button("Скрыть", key=f"hide_details_{report_data['file_index']}"):
                                    st.session_state[f'show_details_{report_data["file_index"]}'] = False
                                    st.rerun()
                            st.write("")
            
            # Массовое скачивание
            if individual_reports:
                st.markdown("#### 📦 Массовое скачивание")
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    download_options = st.multiselect(
                        "Выберите отчеты для скачивания:",
                        options=list(individual_reports.keys()),
                        default=list(individual_reports.keys()),
                        help="Выберите файлы для включения в архив"
                    )
                
                with col2:
                    include_verified = st.checkbox("Включить проверенные", value=True, help="Включить файлы без ошибок")
                
                with col3:
                    only_corrected = st.checkbox("Только исправленные", value=False, help="Только файлы с исправлениями")
                
                if download_options:
                    # Фильтруем по выбранным опциям
                    filtered_reports = {}
                    for file_name in download_options:
                        report_data = individual_reports[file_name]
                        
                        # Применяем фильтры
                        if only_corrected and not report_data['has_errors']:
                            continue
                        if not include_verified and not report_data['has_errors']:
                            continue
                            
                        filtered_reports[file_name] = report_data
                    
                    if filtered_reports:
                        # Создаем ZIP архив
                        zip_buffer = io.BytesIO()
                        
                        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                            for file_name, report_data in filtered_reports.items():
                                try:
                                    if report_data['has_errors']:
                                        # Исправленный файл
                                        corrected_data = report_data['validation']['corrected_data']
                                        export_data = corrected_data[DEF_COLS].copy()
                                        
                                        excel_buffer = io.BytesIO()
                                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                            export_data.to_excel(writer, sheet_name="Corrected Report", index=False)
                                        
                                        zip_file.writestr(f"corrected_{file_name}", excel_buffer.getvalue())
                                    else:
                                        # Проверенный файл
                                        original_data = files_data[report_data['file_index']]
                                        export_data = original_data[DEF_COLS].copy()
                                        
                                        excel_buffer = io.BytesIO()
                                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                            export_data.to_excel(writer, sheet_name="Report", index=False)
                                        
                                        zip_file.writestr(f"verified_{file_name}", excel_buffer.getvalue())
                                        
                                except Exception as e:
                                    st.error(f"Ошибка при добавлении {file_name} в архив: {str(e)}")
                        
                        # Кнопка скачивания архива
                        st.download_button(
                            label=f"📦 Скачать архив ({len(filtered_reports)} файлов)",
                            data=zip_buffer.getvalue(),
                            file_name=f"employee_reports_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                            mime="application/zip",
                            help="Скачать ZIP архив с выбранными отчетами",
                            key="download_reports_archive"
                        )
                    else:
                        st.info("Нет файлов, соответствующих выбранным фильтрам")

st.markdown("---")
st.subheader("Скачать отчёт (Excel)")
if st.button("Сформировать общий отчет"):
    # Автоматическая валидация перед экспортом (используем унифицированную систему)
    validation_result = validate_final_totals(files_data, st.session_state["custom_table"], tolerance=0.01)
    
    if 'error' in validation_result:
        st.warning(f"⚠️ {validation_result['error']}")
    elif not validation_result['is_valid']:
        st.warning("⚠️ Обнаружены расхождения в суммах! Данные автоматически исправлены для экспорта.")
        st.info(f"Разница: {validation_result['difference']:.2f} ₽")
        
        with st.expander("🔍 Показать детали расхождений"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("All tasks (исправлено)", f"{validation_result['all_tasks_total']:.2f} ₽")
            with col2:
                st.metric("For accounting", f"{validation_result['accounting_total']:.2f} ₽")
            
            if validation_result['employee_discrepancies']:
                st.write("**Расхождения по сотрудникам:**")
                for disc in validation_result['employee_discrepancies']:
                    st.write(f"• **{disc['employee']}**: разница {disc['diff']} ₽")
    else:
        st.success("✅ Валидация пройдена! Данные готовы к экспорту.")
    
    import io
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        if files_data:
            # Используем исправленные данные из унифицированной валидации, если доступны
            if 'corrected_data' in validation_result:
                all_tasks = validation_result['corrected_data'].copy()
            else:
                # Fallback: используем стандартную обработку
                all_tasks = pd.concat(files_data, ignore_index=True)
                
                # Убираем дублирование финальных сумм - фильтруем строки с итогами
                if 'ticket' in all_tasks.columns:
                    mask = ~all_tasks['ticket'].astype(str).str.lower().str.contains(
                        'итого|общая сумма|total|сумма:', na=False
                    )
                    all_tasks = all_tasks[mask]
                
                # Также убираем пустые строки с нулевыми значениями
                if 'assignee_name' in all_tasks.columns:
                    mask = (all_tasks['assignee_name'].notna()) & (all_tasks['assignee_name'].str.strip() != '')
                    all_tasks = all_tasks[mask]
                
                # Применяем унифицированные исправления
                for idx, row in all_tasks.iterrows():
                    ticket_validation = validate_ticket_unified(row, tolerance=0.01)
                    if not ticket_validation['is_valid']:
                        for correction in ticket_validation['corrections']:
                            all_tasks.loc[idx, correction['field']] = correction['new_value']
            
            # Фильтруем колонки для экспорта - убираем технические и добавляем нужные
            export_columns = []
            for col in DEF_COLS:
                if col in all_tasks.columns:
                    export_columns.append(col)
            
            # Убираем технические колонки
            tech_columns = ['source_file', 'assignee_name']
            export_columns = [col for col in export_columns if col not in tech_columns]
            
            # Проверяем, есть ли исправленные данные
            if 'corrections_result' in st.session_state and st.session_state['corrections_result']['corrections_summary']['total_corrections'] > 0:
                # Используем исправленные данные
                corrected_data = st.session_state['corrections_result']['corrected_data']
                st.info("✅ Экспортируются исправленные данные с примененными автокоррекциями")
                
                # Фильтруем колонки для экспорта из исправленных данных
                if export_columns:
                    all_tasks_filtered = corrected_data[export_columns] if all(col in corrected_data.columns for col in export_columns) else corrected_data
                else:
                    all_tasks_filtered = corrected_data
            else:
                # Используем оригинальные данные
                if export_columns:
                    all_tasks_filtered = all_tasks[export_columns]
                else:
                    all_tasks_filtered = all_tasks
                
            all_tasks_filtered.to_excel(writer, sheet_name="All tasks", index=False)
            
            # Добавляем лист с отчетом об исправлениях, если они были
            if 'corrections_result' in st.session_state and st.session_state['corrections_result']['corrections_summary']['total_corrections'] > 0:
                corrections_applied = st.session_state['corrections_result']['corrections_applied']
                
                # Создаем DataFrame для отчета об исправлениях
                corrections_df = pd.DataFrame([
                    {
                        'Строка': c['row_number'],
                        'Тикет': c['ticket'],
                        'Исполнитель': c['assignee'],
                        'Поле': c['field'],
                        'Тип_ошибки': c['error_type'],
                        'Было': c['old_value'],
                        'Стало': c['new_value'],
                        'Разница': round(abs(c['new_value'] - c['old_value']), 2),
                        'Описание': c['description'],
                        'Файл': c['source_file']
                    }
                    for c in corrections_applied
                ])
                
                corrections_df.to_excel(writer, sheet_name="Corrections Report", index=False)
        
        # Экспортируем таблицу для бухгалтерии (убираем итоговые строки)
        accounting_table = st.session_state["custom_table"].copy()
        if len(accounting_table) > 0:
            # Убираем итоговые строки из таблицы бухгалтерии
            mask = ~accounting_table["Сотрудник"].str.startswith("📊", na=False)
            accounting_table = accounting_table[mask]
        
        accounting_table.to_excel(writer, sheet_name="For accounting", index=False)
    st.download_button(
        "Скачать billing_summary.xlsx",
        data=output.getvalue(),
        file_name="billing_summary.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ---
# 🆕 БЛОК: Сравнение личного отчета и отчета из выгрузки
st.markdown("---")
st.markdown("### 🔄 Сравнение двух отчетов (личный vs выгрузка)")
st.caption("Загрузите два файла: личный отчет сотрудника и выгрузочный. Корректировки применяются ТОЛЬКО к выгрузке по формулам и данным из личного отчета.")

with st.expander("📄 Загрузить два файла для сравнения", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        personal_file = st.file_uploader("Личный отчет (.xlsx)", type=['xlsx'], key="cmp_personal")
    with c2:
        export_file = st.file_uploader("Отчет из выгрузки (.xlsx)", type=['xlsx'], key="cmp_export")

    if personal_file and export_file:
        try:
            p_df = load_employee_file_from_bytes(personal_file.name, personal_file.read())
            e_df = load_employee_file_from_bytes(export_file.name, export_file.read())

            st.success("Файлы загружены. Готовы к сравнению.")

            if st.button("🔍 Сравнить и скорректировать выгрузку", type="primary", key="cmp_run"):
                cmp_result = compare_and_correct_export(p_df, e_df, tolerance=0.01)
                st.session_state['cmp_result'] = cmp_result

                st.metric("Совпавших тикетов", cmp_result['summary']['matched'])
                st.metric("Строк с изменениями", cmp_result['summary']['changed_rows'])
                st.metric("Всего изменений", cmp_result['summary']['total_changes'])

        except Exception as ex:
            st.error(f"Ошибка обработки файлов: {ex}")

if 'cmp_result' in st.session_state:
    cmp_result = st.session_state['cmp_result']

    with st.expander("📝 Отчет о различиях (по полям)", expanded=True):
        rep = cmp_result['comparison_report']
        if rep is not None and len(rep) > 0:
            st.dataframe(rep, use_container_width=True)
        else:
            st.info("Различий по полям не найдено.")

    with st.expander("❗ Тикеты без пары", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Только в личном отчете")
            st.dataframe(cmp_result['only_in_personal'][['ticket','summary','total']] if len(cmp_result['only_in_personal']) else pd.DataFrame(columns=['ticket','summary','total']), use_container_width=True)
        with col2:
            st.caption("Только в выгрузке")
            st.dataframe(cmp_result['only_in_export'][['ticket','summary','total']] if len(cmp_result['only_in_export']) else pd.DataFrame(columns=['ticket','summary','total']), use_container_width=True)

    # Экспорт
    colx, coly = st.columns([2,1])
    with colx:
        st.caption("Скачать скорректированную выгрузку с отчетом")
    with coly:
        if st.button("💾 Подготовить файл", key="cmp_prepare_export"):
            try:
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Лист 1: Скорректированная выгрузка
                    cmp_result['corrected_export'][DEF_COLS].to_excel(writer, sheet_name="Corrected Export", index=False)

                    # Лист 2: Отчет о различиях
                    rep = cmp_result['comparison_report']
                    (rep if rep is not None and len(rep) > 0 else pd.DataFrame(columns=['ticket','field','export_value','personal_value','new_export_value','change_applied','reason']))\
                        .to_excel(writer, sheet_name="Comparison Report", index=False)

                    # Лист 3: Unmatched
                    unmatched_p = cmp_result['only_in_personal'][['ticket','summary','total']] if len(cmp_result['only_in_personal']) else pd.DataFrame(columns=['ticket','summary','total'])
                    unmatched_e = cmp_result['only_in_export'][['ticket','summary','total']] if len(cmp_result['only_in_export']) else pd.DataFrame(columns=['ticket','summary','total'])

                    unmatched_p.to_excel(writer, sheet_name="Only in Personal", index=False)
                    unmatched_e.to_excel(writer, sheet_name="Only in Export", index=False)

                st.download_button(
                    label="📥 Скачать результат",
                    data=output.getvalue(),
                    file_name=f"comparison_result_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="cmp_download_file"
                )
            except Exception as ex:
                st.error(f"Не удалось подготовить файл: {ex}")

