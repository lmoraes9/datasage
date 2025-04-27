# analysis/views.py
from django.shortcuts import render, redirect
from django.http import JsonResponse, FileResponse, Http404
from django.views.decorators.http import require_POST
import json
import pandas as pd
import numpy as np
import os
import uuid
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import matplotlib
matplotlib.use('Agg') # Backend não-interativo para Matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from django.urls import reverse
import io
import base64
import traceback
import joblib # Para salvar/carregar modelos/pipelines

# --- Imports do Scikit-learn ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Modelos
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Métricas
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc,
    mean_squared_error, r2_score, ConfusionMatrixDisplay, RocCurveDisplay
)

# --- Configurações ---
TEMP_STORAGE_PATH = os.path.join(settings.MEDIA_ROOT, 'temp_uploads')
os.makedirs(TEMP_STORAGE_PATH, exist_ok=True)

# --- Views ---

def home_page(request):
    """Handles file upload or sample data loading."""
    if request.method == 'GET':
        request.session.flush() # Limpa sessão antiga ao visitar a home
    if request.method == 'POST':
        session_id = str(uuid.uuid4())
        request.session['session_id'] = session_id
        temp_dir = os.path.join(TEMP_STORAGE_PATH, session_id)
        os.makedirs(temp_dir, exist_ok=True)

        if 'sample_data' in request.POST:
            sample_filename = 'iris_sample_data.csv'
            file_path = os.path.join(temp_dir, sample_filename)
            try:
                # Tenta carregar Iris do Seaborn
                iris_df = sns.load_dataset('iris')
                iris_df.to_csv(file_path, index=False)
                request.session['data_path'] = file_path
                request.session['original_filename'] = sample_filename
                print(f"[Session {session_id}] Iris sample saved: {file_path}")
                return redirect('analysis:analyze')
            except ImportError:
                return render(request, 'analysis/home.html', {'error': 'Seaborn library not found. Cannot load sample data. Try `pip install seaborn`.'})
            except Exception as e:
                print(f"Error preparing Iris sample: {e}")
                return render(request, 'analysis/home.html', {'error': f'Error preparing sample data: {e}'})
        elif 'data_file' in request.FILES:
            uploaded_file = request.FILES['data_file']
            if not uploaded_file.name.endswith(('.csv', '.xlsx')):
                return render(request, 'analysis/home.html', {'error': 'Invalid file type. Please upload CSV or XLSX.'})
            # Salva o arquivo no diretório temporário da sessão
            fs = FileSystemStorage(location=temp_dir)
            filename = fs.save(uploaded_file.name, uploaded_file)
            file_path = os.path.join(temp_dir, filename)
            request.session['data_path'] = file_path
            request.session['original_filename'] = filename
            print(f"[Session {session_id}] File uploaded and saved: {file_path}")
            return redirect('analysis:analyze')
        else:
             return render(request, 'analysis/home.html', {'error': 'No file uploaded or sample data selected.'})

    # Método GET ou se POST falhar antes do redirect
    return render(request, 'analysis/home.html')


def analysis_page(request):
    """Displays the main analysis interface with detailed stats."""
    session_id = request.session.get('session_id')
    data_path = request.session.get('data_path')
    # Usa o nome mais recente (pode ter sido limpo/capped) ou o original
    current_filename = request.session.get('cleaned_filename', request.session.get('original_filename', 'data file'))

    if not all([session_id, data_path, os.path.exists(data_path)]):
        print("Redirecting home: Missing session/data.")
        request.session.flush()
        return redirect('analysis:home')

    try:
        # --- Carregar Dados (do path atual na sessão) ---
        print(f"[Session {session_id}] Loading data for analysis page from: {data_path}")
        if data_path.endswith('.csv'): df = pd.read_csv(data_path)
        elif data_path.endswith('.xlsx'): df = pd.read_excel(data_path)
        else:
            print(f"Unsupported file type in session: {data_path}")
            request.session.flush(); return redirect('analysis:home')

        is_empty = df.empty
        if is_empty: print(f"Warning: Data file is empty {data_path}")

        # --- Estatísticas Gerais ---
        num_rows = len(df)
        num_cols = len(df.columns) if not is_empty else 0
        missing_values = df.isnull().sum().sum() if not is_empty else 0
        total_cells = df.size
        missing_percentage = round((missing_values / total_cells) * 100, 2) if total_cells > 0 else 0
        duplicate_rows = df.duplicated().sum() if not is_empty else 0
        all_columns_list = list(df.columns) if not is_empty else []
        numeric_cols_list = df.select_dtypes(include=np.number).columns.tolist() if not is_empty else []
        categorical_cols_list = df.select_dtypes(include=['object', 'category', 'boolean']).columns.tolist() if not is_empty else []
        sample_data_html = df.head().to_html(
            classes="table table-sm table-striped table-bordered table-dark table-hover", # Adiciona as classes desejadas
            border=0, # Remove a borda padrão do to_html
            index=True # Mantém o índice
        ) if not is_empty else "<p class='text-warning'>Data is empty.</p>"

        # --- Cálculo Detalhado de Estatísticas por Coluna ---
        column_details = {}
        if not is_empty:
            for col in all_columns_list:
                col_data = df[col]
                details = {'name': col, 'missing': col_data.isnull().sum()}
                details['missing_pct'] = round((details['missing'] / num_rows) * 100, 2) if num_rows > 0 else 0
                details['unique_count'] = col_data.nunique()

                if col in numeric_cols_list:
                    desc = col_data.describe()
                    details.update({ 'type': 'Numeric', 'mean': desc.get('mean'), 'std': desc.get('std'), 'min': desc.get('min'), 'q1': desc.get('25%'), 'median': desc.get('50%'), 'q3': desc.get('75%'), 'max': desc.get('max'), 'iqr': desc.get('75%', np.nan) - desc.get('25%', np.nan), 'skew': col_data.skew(), 'kurt': col_data.kurt(), })
                else: # Inclui object, category, boolean
                    desc = col_data.describe(include='all') # Usar 'all' para pegar count, unique, top, freq
                    mode_val = col_data.mode()
                    details.update({ 'type': str(col_data.dtype), 'count': desc.get('count'), 'top': desc.get('top', mode_val[0] if not mode_val.empty else 'N/A'), 'freq': desc.get('freq', col_data.value_counts().iloc[0] if not col_data.value_counts().empty else 0), })

                column_details[col] = details # Arredondar aqui se necessário

        # --- Preparar Contexto ---
        context = {
            'filename': current_filename, # Mostrar nome do arquivo atual
            'num_rows': num_rows, 'num_cols': num_cols, 'missing_percentage': missing_percentage,
            'duplicate_rows': duplicate_rows, 'sample_data_html': sample_data_html,
            'columns': all_columns_list, 'numeric_columns': numeric_cols_list,
            'categorical_columns': categorical_cols_list, 'is_empty': is_empty,
            'column_details': column_details,
        }
        print(f"[Session {session_id}] Rendering analysis page with {len(column_details)} column details.")
        return render(request, 'analysis/analysis_page.html', context)

    except pd.errors.EmptyDataError:
        print(f"Error loading data (EmptyDataError): {data_path}")
        request.session.flush(); return redirect('analysis:home')
    except FileNotFoundError:
         print(f"Error loading data (FileNotFoundError): {data_path}")
         request.session.flush(); return redirect('analysis:home')
    except Exception as e:
        print(f"CRITICAL Error in analysis_page: {e}\n{traceback.format_exc()}")
        request.session.flush(); return redirect('analysis:home')


@require_POST
def ajax_get_correlation(request):
    """Generates correlation heatmap."""
    session_id = request.session.get('session_id'); data_path = request.session.get('data_path'); temp_dir = os.path.join(TEMP_STORAGE_PATH, session_id) if session_id else None
    if not all([session_id, data_path, os.path.exists(data_path), temp_dir]): return JsonResponse({'status': 'error', 'error': 'Session/Data expired.'}, status=400)
    try:
        body = json.loads(request.body); selected_columns = body.get('selected_columns', [])
        if not isinstance(selected_columns, list) or len(selected_columns) < 2: return JsonResponse({'status': 'error', 'error': 'Please select at least 2 columns.'}, status=400)
        if data_path.endswith('.csv'): df = pd.read_csv(data_path)
        elif data_path.endswith('.xlsx'): df = pd.read_excel(data_path)
        else: return JsonResponse({'status': 'error', 'error': 'Invalid file type.'}, status=400)

        numeric_cols_in_df = df.select_dtypes(include=np.number).columns
        valid_selected_cols = [col for col in selected_columns if col in numeric_cols_in_df]
        if len(valid_selected_cols) < 2: return JsonResponse({'status': 'error', 'error': 'Please select at least 2 NUMERIC columns found in the data.'}, status=400)

        correlation_matrix = df[valid_selected_cols].corr()
        plt.figure(figsize=(min(max(8, len(valid_selected_cols)), 15), min(max(6, len(valid_selected_cols)), 12))) # Tamanho dinâmico
        sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
        plt.title('Correlation Matrix', fontsize=16); plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0); plt.tight_layout()
        image_filename = f'correlation_{uuid.uuid4().hex[:6]}.png'; image_path = os.path.join(temp_dir, image_filename); plt.savefig(image_path); plt.close()
        correlation_image_url = f"{settings.MEDIA_URL}temp_uploads/{session_id}/{image_filename}";
        print(f"[Session {session_id}] Correlation plot generated: {image_filename}")
        return JsonResponse({'status': 'success', 'image_url': correlation_image_url})
    except json.JSONDecodeError: return JsonResponse({'status': 'error', 'error': 'Invalid request format.'}, status=400)
    except Exception as e: print(f"Error in ajax_get_correlation: {e}\n{traceback.format_exc()}"); return JsonResponse({'status': 'error', 'error': f'Error generating correlation plot: {e}'}, status=500)

@require_POST
def ajax_get_scatter_plot(request):
    """Generates scatter plot."""
    session_id = request.session.get('session_id'); data_path = request.session.get('data_path'); temp_dir = os.path.join(TEMP_STORAGE_PATH, session_id) if session_id else None
    if not all([session_id, data_path, os.path.exists(data_path), temp_dir]): return JsonResponse({'status': 'error', 'error': 'Session/Data expired.'}, status=400)
    try:
        body = json.loads(request.body); x_col = body.get('x_column'); y_col = body.get('y_column')
        if not x_col or not y_col: return JsonResponse({'status': 'error', 'error': 'Please select both X and Y variables.'}, status=400)
        if x_col == y_col: return JsonResponse({'status': 'error', 'error': 'X and Y variables cannot be the same.'}, status=400)

        if data_path.endswith('.csv'): df = pd.read_csv(data_path)
        elif data_path.endswith('.xlsx'): df = pd.read_excel(data_path)
        else: return JsonResponse({'status': 'error', 'error': 'Invalid data file type.'}, status=400)
        if x_col not in df.columns or y_col not in df.columns: return JsonResponse({'status': 'error', 'error': f'Selected column(s) not found: {x_col if x_col not in df else ""}{"," if x_col not in df and y_col not in df else ""} {y_col if y_col not in df else ""}'}, status=400)

        numeric_cols = df.select_dtypes(include=np.number).columns
        if x_col not in numeric_cols or y_col not in numeric_cols: return JsonResponse({'status': 'error', 'error': 'Scatter plots require NUMERIC columns for both X and Y axes.'}, status=400)

        plt.figure(figsize=(10, 6)); sns.scatterplot(data=df, x=x_col, y=y_col, alpha=0.7)
        plt.title(f'Scatter Plot: {y_col} vs {x_col}', fontsize=16); plt.xlabel(x_col, fontsize=12); plt.ylabel(y_col, fontsize=12); plt.grid(True, linestyle='--', alpha=0.5); plt.tight_layout()
        safe_x = "".join(c if c.isalnum() else '_' for c in x_col)
        safe_y = "".join(c if c.isalnum() else '_' for c in y_col)
        image_filename = f'scatter_{safe_y}_vs_{safe_x}_{uuid.uuid4().hex[:6]}.png'; image_path = os.path.join(temp_dir, image_filename); plt.savefig(image_path); plt.close()
        scatter_image_url = f"{settings.MEDIA_URL}temp_uploads/{session_id}/{image_filename}";
        print(f"[Session {session_id}] Scatter plot generated: {image_filename}")
        return JsonResponse({'status': 'success', 'image_url': scatter_image_url, 'x_col': x_col, 'y_col': y_col})
    except json.JSONDecodeError: return JsonResponse({'status': 'error', 'error': 'Invalid request format.'}, status=400)
    except Exception as e: print(f"Error in ajax_get_scatter_plot: {e}\n{traceback.format_exc()}"); return JsonResponse({'status': 'error', 'error': f'Error generating scatter plot: {e}'}, status=500)


@require_POST
def ajax_get_univariate_plot(request):
    """Generates and returns URL for a univariate plot (hist/box or bar)."""
    session_id = request.session.get('session_id'); data_path = request.session.get('data_path'); temp_dir = os.path.join(TEMP_STORAGE_PATH, session_id) if session_id else None
    if not all([session_id, data_path, os.path.exists(data_path), temp_dir]): return JsonResponse({'status': 'error', 'error': 'Session/Data expired.'}, status=400)
    try:
        body = json.loads(request.body); column_name = body.get('column_name')
        if not column_name: return JsonResponse({'status': 'error', 'error': 'Column name not provided.'}, status=400)

        if data_path.endswith('.csv'): df = pd.read_csv(data_path)
        elif data_path.endswith('.xlsx'): df = pd.read_excel(data_path)
        else: return JsonResponse({'status': 'error', 'error': 'Invalid data file type.'}, status=400)
        if column_name not in df.columns: return JsonResponse({'status': 'error', 'error': f'Column "{column_name}" not found.'}, status=400)
        if df[column_name].isnull().all(): return JsonResponse({'status': 'error', 'error': f'Column "{column_name}" contains only missing values.'}, status=400)

        image_urls = {}; safe_col_name = "".join(c if c.isalnum() else '_' for c in column_name)
        col_data = df[column_name].dropna()
        plt.style.use('seaborn-v0_8-darkgrid')

        if pd.api.types.is_numeric_dtype(df[column_name]):
            fig_hist, ax_hist = plt.subplots(figsize=(7, 4)); sns.histplot(col_data, kde=True, ax=ax_hist, color='skyblue'); ax_hist.set_title(f'Distribution of {column_name}', fontsize=12); ax_hist.set_xlabel(column_name, fontsize=10); ax_hist.set_ylabel('Frequency', fontsize=10); plt.tight_layout(); hist_filename = f'hist_{safe_col_name}_{uuid.uuid4().hex[:6]}.png'; hist_path = os.path.join(temp_dir, hist_filename); plt.savefig(hist_path); plt.close(fig_hist); image_urls['histogram'] = f"{settings.MEDIA_URL}temp_uploads/{session_id}/{hist_filename}"; print(f"[Session {session_id}] Histogram generated: {hist_filename}")
            fig_box, ax_box = plt.subplots(figsize=(7, 2.5)); sns.boxplot(x=col_data, ax=ax_box, color='lightcoral'); ax_box.set_title(f'Box Plot of {column_name}', fontsize=12); ax_box.set_xlabel(column_name, fontsize=10); plt.tight_layout(); box_filename = f'box_{safe_col_name}_{uuid.uuid4().hex[:6]}.png'; box_path = os.path.join(temp_dir, box_filename); plt.savefig(box_path); plt.close(fig_box); image_urls['boxplot'] = f"{settings.MEDIA_URL}temp_uploads/{session_id}/{box_filename}"; print(f"[Session {session_id}] Boxplot generated: {box_filename}")
        elif pd.api.types.is_categorical_dtype(df[column_name]) or pd.api.types.is_object_dtype(df[column_name]) or pd.api.types.is_bool_dtype(df[column_name]):
             max_categories = 20; value_counts = col_data.value_counts();
             if len(value_counts) > max_categories: top_categories = value_counts.nlargest(max_categories); other_count = value_counts.nsmallest(len(value_counts) - max_categories).sum(); top_categories['Other'] = other_count if other_count > 0 else 0; plot_data = top_categories.loc[top_categories > 0]; plot_title = f'Top {max_categories} Categories of {column_name}'
             else: plot_data = value_counts; plot_title = f'Counts per Category in {column_name}'
             if not plot_data.empty:
                 fig_bar, ax_bar = plt.subplots(figsize=(7, max(4, len(plot_data) * 0.4))); sns.barplot(x=plot_data.values, y=plot_data.index.astype(str), ax=ax_bar, palette='viridis', orient='h'); ax_bar.set_title(plot_title, fontsize=12); ax_bar.set_xlabel('Count', fontsize=10); ax_bar.set_ylabel('Category', fontsize=10); plt.tight_layout(); bar_filename = f'bar_{safe_col_name}_{uuid.uuid4().hex[:6]}.png'; bar_path = os.path.join(temp_dir, bar_filename); plt.savefig(bar_path); plt.close(fig_bar); image_urls['barchart'] = f"{settings.MEDIA_URL}temp_uploads/{session_id}/{bar_filename}"; print(f"[Session {session_id}] Barchart generated: {bar_filename}")
             else: return JsonResponse({'status': 'error', 'error': f'No data to plot for column "{column_name}".'}, status=400)
        else: return JsonResponse({'status': 'error', 'error': f'Unsupported data type for plotting: {df[column_name].dtype}'}, status=400)

        if not image_urls: return JsonResponse({'status': 'error', 'error': 'Could not generate any plot.'}, status=500)
        return JsonResponse({'status': 'success', 'image_urls': image_urls, 'column': column_name})
    except json.JSONDecodeError: return JsonResponse({'status': 'error', 'error': 'Invalid request format.'}, status=400)
    except Exception as e: print(f"Error in ajax_get_univariate_plot: {e}\n{traceback.format_exc()}"); return JsonResponse({'status': 'error', 'error': f'Error generating plot for {body.get("column_name", "column")}: {e}'}, status=500)


@require_POST
def ajax_apply_cleaning(request):
    """Applies Missing Value handling and Duplicate Removal."""
    session_id = request.session.get('session_id'); current_data_path = request.session.get('data_path'); original_filename = request.session.get('original_filename', 'data'); temp_dir = os.path.join(TEMP_STORAGE_PATH, session_id) if session_id else None
    if not all([session_id, current_data_path, os.path.exists(current_data_path), temp_dir]): return JsonResponse({'status': 'error', 'error': 'Session/Data expired.'}, status=400)
    try:
        options = json.loads(request.body); missing_strategy = options.get('missing_strategy', 'none'); remove_duplicates = options.get('remove_duplicates', False); constant_value = options.get('missing_constant_value', None)
        print(f"[Session {session_id}] Cleaning options received: {options}")

        try:
            if current_data_path.endswith('.csv'): df = pd.read_csv(current_data_path)
            elif current_data_path.endswith('.xlsx'): df = pd.read_excel(current_data_path)
            else: return JsonResponse({'status': 'error', 'error': 'Invalid data file type.'}, status=400)
        except Exception as e: print(f"Error loading data for cleaning: {e}"); return JsonResponse({'status': 'error', 'error': f'Error loading data: {e}'}, status=500)

        original_rows = len(df); df_cleaned = df.copy(); rows_affected_missing = 0; rows_affected_duplicates = 0

        # --- Aplicar Limpeza ---
        if missing_strategy == 'remove_rows':
            rows_before = len(df_cleaned)
            df_cleaned.dropna(inplace=True)
            rows_affected_missing = rows_before - len(df_cleaned)
        elif missing_strategy in ['fill_mean', 'fill_median']:
            num_cols = df_cleaned.select_dtypes(include=np.number).columns
            if not num_cols.empty:
                fill_values = df_cleaned[num_cols].mean() if missing_strategy == 'fill_mean' else df_cleaned[num_cols].median()
                # Aplicar fillna apenas onde os valores de preenchimento não são NaN
                for col in num_cols:
                    if pd.notna(fill_values[col]):
                        df_cleaned[col].fillna(fill_values[col], inplace=True)
                    else:
                         print(f"Warning: Cannot fill NaNs in column '{col}' because calculated {missing_strategy} is also NaN.")
        elif missing_strategy == 'fill_mode':
            for col in df_cleaned.columns:
                if df_cleaned[col].isnull().any():
                    mode_val = df_cleaned[col].mode()
                    if not mode_val.empty:
                         df_cleaned[col].fillna(mode_val[0], inplace=True)
        elif missing_strategy == 'fill_constant':
             if constant_value is None or str(constant_value).strip() == '': return JsonResponse({'status': 'error', 'error': 'Constant value cannot be empty when using fill_constant strategy.'}, status=400)
             # Tentar converter para numérico se possível, senão usar como string
             try: fill_val = pd.to_numeric(constant_value)
             except ValueError: fill_val = str(constant_value)
             df_cleaned.fillna(fill_val, inplace=True)

        if remove_duplicates:
            rows_before = len(df_cleaned)
            df_cleaned.drop_duplicates(inplace=True)
            rows_affected_duplicates = rows_before - len(df_cleaned)

        total_rows_affected = rows_affected_missing + rows_affected_duplicates
        message = f"Cleaning applied. Missing Values: {rows_affected_missing} rows removed/filled. Duplicates: {rows_affected_duplicates} rows removed."

        # --- Salvar se houve mudanças ---
        # Verifica se o DataFrame mudou (comparação pode ser pesada, mas mais segura)
        # Ou simplificar: salvar sempre que total_rows_affected > 0 ou estratégia de fill foi usada
        needs_saving = total_rows_affected > 0 or missing_strategy.startswith('fill_')

        if needs_saving:
            base_name, ext = os.path.splitext(os.path.basename(current_data_path))
            # Limpa sufixos anteriores para evitar _cleaned_cleaned etc.
            base_name = base_name.replace('_cleaned','').replace('_capped','').replace('_converted','')
            cleaned_filename = f"{base_name}_cleaned{ext}"
            cleaned_file_path = os.path.join(temp_dir, cleaned_filename)
            print(f"[Session {session_id}] Saving cleaned data: {cleaned_file_path}")
            if cleaned_file_path.endswith('.csv'): df_cleaned.to_csv(cleaned_file_path, index=False)
            elif cleaned_file_path.endswith('.xlsx'): df_cleaned.to_excel(cleaned_file_path, index=False)

            request.session['data_path'] = cleaned_file_path
            request.session['cleaned_filename'] = cleaned_filename
            download_url = f"{settings.MEDIA_URL}temp_uploads/{session_id}/{cleaned_filename}"
            new_filename_display = cleaned_filename
        else:
             message = "No changes applied (no missing values action taken and no duplicates found/removed)."
             download_url = None # Não gera nova URL se nada mudou
             new_filename_display = os.path.basename(current_data_path)


        return JsonResponse({ 'status': 'success', 'message': message, 'download_url': download_url, 'cleaned_filename': new_filename_display, # Para UI
                              'new_filename': new_filename_display, # Para JS
                              'rows_removed': total_rows_affected, }) # Renomeado para clareza

    except json.JSONDecodeError: return JsonResponse({'status': 'error', 'error': 'Invalid request format.'}, status=400)
    except Exception as e: print(f"Error during cleaning: {e}\n{traceback.format_exc()}"); return JsonResponse({'status': 'error', 'error': f'Error during data cleaning: {e}'}, status=500)


@require_POST
def ajax_handle_outliers(request):
    """Detects or Caps outliers using IQR method."""
    session_id = request.session.get('session_id'); data_path = request.session.get('data_path'); original_filename = request.session.get('original_filename', 'data'); temp_dir = os.path.join(TEMP_STORAGE_PATH, session_id) if session_id else None
    if not all([session_id, data_path, os.path.exists(data_path), temp_dir]): return JsonResponse({'status': 'error', 'error': 'Session/Data expired.'}, status=400)
    try:
        body = json.loads(request.body); selected_columns = body.get('columns', []); method = body.get('method', 'iqr'); action = body.get('action', 'detect'); iqr_multiplier = float(body.get('iqr_multiplier', 1.5))
        if not selected_columns: return JsonResponse({'status': 'error', 'error': 'Please select numeric columns.'}, status=400)
        if action not in ['detect', 'cap']: return JsonResponse({'status': 'error', 'error': 'Invalid action.'}, status=400)
        if method != 'iqr': return JsonResponse({'status': 'error', 'error': 'Only IQR method supported.'}, status=400)

        if data_path.endswith('.csv'): df = pd.read_csv(data_path)
        elif data_path.endswith('.xlsx'): df = pd.read_excel(data_path)
        else: return JsonResponse({'status': 'error', 'error': 'Invalid data file type.'}, status=400)

        numeric_cols_in_df = df.select_dtypes(include=np.number).columns
        valid_columns = [col for col in selected_columns if col in numeric_cols_in_df]
        if not valid_columns: return JsonResponse({'status': 'error', 'error': 'No valid numeric columns selected.'}, status=400)

        outlier_info = {}; df_modified = df.copy(); rows_affected_total = 0

        for col in valid_columns:
            col_data = df_modified[col].dropna()
            if col_data.empty: outlier_info[col] = {'count': 0, 'percentage': 0.0, 'lower_bound': None, 'upper_bound': None, 'rows_capped': 0}; continue
            Q1 = col_data.quantile(0.25); Q3 = col_data.quantile(0.75); IQR = Q3 - Q1
            if IQR == 0: # Evita divisão por zero ou limites iguais se a coluna for constante
                lower_bound = Q1; upper_bound = Q3;
            else:
                lower_bound = Q1 - iqr_multiplier * IQR; upper_bound = Q3 + iqr_multiplier * IQR

            outliers_mask = (df_modified[col] < lower_bound) | (df_modified[col] > upper_bound)
            outlier_count = outliers_mask.sum() # Conta apenas não-NaNs que são outliers
            outlier_percentage = round((outlier_count / len(col_data)) * 100, 2) if len(col_data) > 0 else 0.0

            rows_capped = 0
            if action == 'cap' and outlier_count > 0:
                df_modified[col] = df_modified[col].clip(lower=lower_bound, upper=upper_bound)
                rows_capped = outlier_count # clip aplica a ambos os limites

            outlier_info[col] = {'count': outlier_count, 'percentage': outlier_percentage, 'lower_bound': round(lower_bound, 4) if pd.notna(lower_bound) else None, 'upper_bound': round(upper_bound, 4) if pd.notna(upper_bound) else None, 'rows_capped': rows_capped}
            rows_affected_total += rows_capped

        download_url = None; capped_filename = None
        if action == 'cap' and rows_affected_total > 0:
            base_name, ext = os.path.splitext(os.path.basename(data_path))
            base_name = base_name.replace('_cleaned','').replace('_capped','').replace('_converted','') # Limpa sufixos
            capped_filename = f"{base_name}_capped{ext}"
            capped_file_path = os.path.join(temp_dir, capped_filename)
            print(f"[Session {session_id}] Saving outlier capped data: {capped_file_path}")
            if capped_file_path.endswith('.csv'): df_modified.to_csv(capped_file_path, index=False)
            elif capped_file_path.endswith('.xlsx'): df_modified.to_excel(capped_file_path, index=False)
            request.session['data_path'] = capped_file_path
            request.session['cleaned_filename'] = capped_filename
            download_url = f"{settings.MEDIA_URL}temp_uploads/{session_id}/{capped_filename}"

        return JsonResponse({'status': 'success', 'action': action, 'outlier_info': outlier_info, 'rows_affected_total': rows_affected_total, 'download_url': download_url, 'new_filename': capped_filename if action == 'cap' and rows_affected_total > 0 else os.path.basename(data_path)})
    except json.JSONDecodeError: return JsonResponse({'status': 'error', 'error': 'Invalid request format.'}, status=400)
    except Exception as e: print(f"Error in ajax_handle_outliers: {e}\n{traceback.format_exc()}"); return JsonResponse({'status': 'error', 'error': f'Error handling outliers: {e}'}, status=500)


# View ajax_convert_data_type REMOVIDA

@require_POST
def ajax_train_multiple_models(request):
    """Trains multiple models, returns results table data and pipeline download links."""
    session_id = request.session.get('session_id'); current_data_path = request.session.get('data_path'); temp_dir = os.path.join(TEMP_STORAGE_PATH, session_id) if session_id else None
    if not all([session_id, current_data_path, os.path.exists(current_data_path), temp_dir]): return JsonResponse({'status': 'error', 'error': 'Session/Data expired or invalid.'}, status=400)
    try:
        params = json.loads(request.body); target_variable = params.get('target_variable'); selected_model_names = params.get('selected_models', []); test_size = float(params.get('test_size', 0.2))
        print(f"\n[Session {session_id}] --- Starting Multiple Model Training --- Target: {target_variable}, Models: {selected_model_names}, Test Size: {test_size}")
        if not target_variable: return JsonResponse({'status': 'error', 'error': 'Target variable is required.'}, status=400)
        if not selected_model_names: return JsonResponse({'status': 'error', 'error': 'Please select at least one model.'}, status=400)
        if not isinstance(selected_model_names, list): return JsonResponse({'status': 'error', 'error': 'Invalid format for selected models.'}, status=400)

        # --- Carregar Dados ---
        try:
            if current_data_path.endswith('.csv'): df = pd.read_csv(current_data_path)
            elif current_data_path.endswith('.xlsx'): df = pd.read_excel(current_data_path)
            else: return JsonResponse({'status': 'error', 'error': 'Invalid data file type.'}, status=400)
            if df.empty: return JsonResponse({'status': 'error', 'error': 'Data file is empty.'}, status=400)
            if target_variable not in df.columns: return JsonResponse({'status': 'error', 'error': f'Target column "{target_variable}" not found.'}, status=400)
        except Exception as e: print(f"Error loading data: {e}"); return JsonResponse({'status': 'error', 'error': f'Error loading data: {e}'}, status=500)

        # --- Pré-processamento Comum ---
        df_filtered = df.copy().dropna(subset=[target_variable])
        if df_filtered.empty: return JsonResponse({'status': 'error', 'error': 'No data left after removing rows with missing target.'}, status=400)
        y_full = df_filtered[target_variable]; X_full = df_filtered.drop(columns=[target_variable])

        target_col_check = y_full.dropna(); target_dtype = target_col_check.dtype; unique_values_count = target_col_check.nunique(); problem_type = 'unsupported'
        if pd.api.types.is_numeric_dtype(target_dtype): problem_type = 'classification' if unique_values_count <= 15 else 'regression'
        elif pd.api.types.is_string_dtype(target_dtype) or pd.api.types.is_categorical_dtype(target_dtype) or pd.api.types.is_object_dtype(target_dtype) or pd.api.types.is_bool_dtype(target_dtype): problem_type = 'classification' if unique_values_count > 1 else 'unsuitable'
        if problem_type in ['unsuitable', 'unsupported']: return JsonResponse({'status': 'error', 'error': f'Target type ({target_dtype}) / unique values ({unique_values_count}) unsuitable.'}, status=400)
        print(f"  Problem Type: {problem_type}")

        label_encoder = None; y_processed = y_full.copy()
        if problem_type == 'classification' and not pd.api.types.is_numeric_dtype(y_processed.dtype):
            label_encoder = LabelEncoder(); y_processed = label_encoder.fit_transform(y_processed); print(f"  Target classes encoded: {list(label_encoder.classes_)}")

        numeric_features = X_full.select_dtypes(include=np.number).columns.tolist(); categorical_features = X_full.select_dtypes(exclude=np.number).columns.tolist(); print(f"  Num Features: {numeric_features}"); print(f"  Cat Features: {categorical_features}")
        transformers_list = []
        if numeric_features: transformers_list.append(('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features))
        if categorical_features: transformers_list.append(('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features))
        if not transformers_list: return JsonResponse({'status': 'error', 'error': 'No features found to build model.'}, status=400)
        preprocessor = ColumnTransformer(transformers=transformers_list, remainder='drop'); print("  Preprocessor defined.")

        print(f"  Splitting data (Test Size: {test_size})..."); X_train, X_test, y_train, y_test = train_test_split( X_full, y_processed, test_size=test_size, random_state=42, stratify=(y_processed if problem_type == 'classification' and len(np.unique(y_processed)) > 1 else None) )
        print(f"  Train shapes: X={X_train.shape} | Test shapes: X={X_test.shape}")

        # --- Mapeamento de Modelos ---
        model_map = { 'LinearRegression': {'model': LinearRegression, 'type': 'regression'}, 'DecisionTreeRegressor': {'model': DecisionTreeRegressor, 'type': 'regression', 'display_name': 'DecisionTree'}, 'RandomForestRegressor': {'model': RandomForestRegressor, 'type': 'regression', 'display_name': 'RandomForest'}, 'LogisticRegression': {'model': LogisticRegression, 'type': 'classification'}, 'DecisionTreeClassifier': {'model': DecisionTreeClassifier, 'type': 'classification', 'display_name': 'DecisionTree'}, 'RandomForestClassifier': {'model': RandomForestClassifier, 'type': 'classification', 'display_name': 'RandomForest'}, 'SVC': {'model': SVC, 'type': 'classification', 'probability': True, 'display_name': 'SVM'}, 'KNeighborsClassifier': {'model': KNeighborsClassifier, 'type': 'classification', 'display_name': 'KNN'} }

        # --- Loop de Treinamento ---
        all_results = {}
        for model_key_name in selected_model_names:
            print(f"\n  --- Training Model: {model_key_name} ---")
            model_results = {'status': 'Skipped', 'metrics': {}, 'plots': {}, 'download_url': None, 'pipeline_filename': None, 'error': None}
            model_entry = None; class_name_used = None; display_name = model_key_name

            # Encontra entrada no model_map
            if model_key_name in model_map:
                 if model_map[model_key_name]['type'] == problem_type: model_entry = model_map[model_key_name]; class_name_used = model_key_name
                 else: model_results['status'] = 'Failed'; model_results['error'] = f"Unsuitable for {problem_type}."; all_results[display_name] = model_results; print(f"    Skipping {model_key_name}: Unsuitable type."); continue
            else: # Tenta pelo display_name
                 found = False;
                 for key, value in model_map.items():
                     if value.get('display_name') == model_key_name and value['type'] == problem_type: model_entry = value; class_name_used = key; display_name = value.get('display_name'); found = True; break
                 if not found: model_results['status'] = 'Failed'; model_results['error'] = f"Model '{model_key_name}' not found/unsuitable."; all_results[model_key_name] = model_results; print(f"    Skipping {model_key_name}: Not found/unsuitable."); continue
            if not model_entry or not class_name_used: model_results['status'] = 'Failed'; model_results['error'] = 'Internal config error.'; all_results[display_name] = model_results; print(f"    Skipping {model_key_name}: Internal lookup error."); continue

            try:
                # Instanciar Modelo
                model_params = {'random_state': 42} if 'random_state' in model_entry['model']().get_params() else {}
                if model_entry.get('probability'): model_params['probability'] = True
                try: model = model_entry['model'](**model_params)
                except TypeError: model_params.pop('random_state', None); model = model_entry['model'](**model_params)
                print(f"    Model {class_name_used} instantiated.")

                # Criar e Treinar Pipeline
                final_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)]); print(f"    Fitting pipeline..."); final_pipeline.fit(X_train, y_train); print("    Fit complete.")

                # Previsões
                print("    Predicting..."); y_pred = final_pipeline.predict(X_test); y_pred_proba = None
                if problem_type == 'classification' and hasattr(final_pipeline, "predict_proba"):
                    try: y_pred_proba = final_pipeline.predict_proba(X_test); print("    Probs generated.")
                    except Exception as e_proba: print(f"    Warn: predict_proba failed: {e_proba}")

                # Métricas
                print("    Calculating metrics..."); metrics = {'problem_type': problem_type}
                if problem_type == 'classification':
                    avg = 'macro' if len(np.unique(y_test)) > 2 else 'binary'
                    metrics['accuracy'] = accuracy_score(y_test, y_pred); metrics['precision'] = precision_score(y_test, y_pred, average=avg, zero_division=0); metrics['recall'] = recall_score(y_test, y_pred, average=avg, zero_division=0); metrics['f1_score'] = f1_score(y_test, y_pred, average=avg, zero_division=0);
                    if y_pred_proba is not None and len(np.unique(y_test)) == 2:
                         try: fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1]); metrics['roc_auc'] = auc(fpr, tpr)
                         except Exception as e_auc: print(f"    Warn: ROC AUC calculation failed: {e_auc}")
                elif problem_type == 'regression': metrics['r2_score'] = r2_score(y_test, y_pred); metrics['mean_squared_error'] = mean_squared_error(y_test, y_pred); metrics['rmse'] = np.sqrt(metrics['mean_squared_error'])
                # Arredonda métricas para exibição (opcional, pode ser feito no JS também)
                metrics = {k: round(v, 4) if isinstance(v, (float, np.number)) else v for k, v in metrics.items()}
                model_results['metrics'] = metrics

                # Salvar Pipeline
                print("    Saving trained pipeline..."); pipeline_filename = f"pipeline_{display_name}_{target_variable}_{uuid.uuid4().hex[:6]}.joblib"; pipeline_filename = "".join(c if c.isalnum() or c in ['_', '.'] else '_' for c in pipeline_filename); pipeline_path = os.path.join(temp_dir, pipeline_filename)
                try: joblib.dump(final_pipeline, pipeline_path); model_results['download_url'] = f"{settings.MEDIA_URL}temp_uploads/{session_id}/{pipeline_filename}"; model_results['pipeline_filename'] = pipeline_filename; print(f"    Pipeline saved: {pipeline_path}")
                except Exception as e_save: print(f"    ERROR saving pipeline: {e_save}"); model_results['error'] = "Failed to save pipeline."; model_results['download_url'] = None; model_results['pipeline_filename'] = None

                model_results['status'] = 'Success'
            except Exception as model_error: print(f"    ERROR Training {display_name}: {model_error}\n{traceback.format_exc()}"); model_results['status'] = 'Failed'; model_results['error'] = f"Training failed: {model_error}"
            all_results[display_name] = model_results # Usa display_name como chave

        print(f"[Session {session_id}] --- Multiple Model Training Complete ---")
        return JsonResponse({'status': 'success', 'message': 'Finished training selected models.', 'results': all_results, 'target_variable': target_variable, 'problem_type': problem_type})
    except json.JSONDecodeError: return JsonResponse({'status': 'error', 'error': 'Invalid request format.'}, status=400)
    except pd.errors.EmptyDataError: return JsonResponse({'status': 'error', 'error': 'Cannot train models: data file empty.'}, status=400)
    except KeyError as e: print(f"KeyError: {e}\n{traceback.format_exc()}"); return JsonResponse({'status': 'error', 'error': f'Column not found: {e}.'}, status=400)
    except ValueError as e: print(f"ValueError: {e}\n{traceback.format_exc()}"); return JsonResponse({'status': 'error', 'error': f'Data/parameter error: {e}'}, status=400)
    except Exception as e: print(f"Unexpected Training Error: {e}\n{traceback.format_exc()}"); return JsonResponse({'status': 'error', 'error': 'Internal server error during training. Check logs.'}, status=500)


# --- Views de Download (se necessário remover no futuro) ---
# Se decidir remover downloads diretos de arquivos, estas podem ser removidas.
# Por enquanto, são necessárias para o download do pipeline .joblib
# def download_file(request, session_id, filename):
#     # ... (implementação segura para download, verificando session_id e filename)
#     # Lembre-se de sanitizar 'filename' e verificar se pertence à sessão!
#     file_path = os.path.join(TEMP_STORAGE_PATH, session_id, filename)
#     if os.path.exists(file_path) and request.session.session_key == session_id: # Exemplo de verificação
#         try:
#             return FileResponse(open(file_path, 'rb'), as_attachment=True, filename=filename)
#         except FileNotFoundError:
#             raise Http404("File not found")
#         except Exception as e:
#              print(f"Error downloading file {filename} for session {session_id}: {e}")
#              raise Http404("Error accessing file")
#     else:
#          raise Http404("File not found or access denied")