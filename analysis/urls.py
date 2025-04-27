# analysis/urls.py
from django.urls import path
from . import views

app_name = 'analysis'

urlpatterns = [
    path('', views.home_page, name='home'),
    path('analyze/', views.analysis_page, name='analyze'),

    # --- URLs AJAX ---
    path('ajax/get_correlation/', views.ajax_get_correlation, name='ajax_get_correlation'),
    path('ajax/apply_cleaning/', views.ajax_apply_cleaning, name='ajax_apply_cleaning'), # Para Missing/Duplicates
    path('ajax/get_scatter_plot/', views.ajax_get_scatter_plot, name='ajax_get_scatter_plot'),
    path('ajax/get_univariate_plot/', views.ajax_get_univariate_plot, name='ajax_get_univariate_plot'),
    path('ajax/handle_outliers/', views.ajax_handle_outliers, name='ajax_handle_outliers'), # Para Outliers
    # path('ajax/train_model/', views.ajax_train_model, name='ajax_train_model'), # Pode remover se não quiser mais treino único
    path('ajax/train_multiple_models/', views.ajax_train_multiple_models, name='ajax_train_multiple_models'), # Para múltiplos modelos

    # --- URLs de Download ---
    # Adicionar rota para download seguro se necessário, ex:
    # path('download/<str:session_id>/<path:filename>/', views.download_file, name='download_file'),
    # NOTA: A implementação de `views.download_file` precisa ser segura!
]