from django.urls import path  # type: ignore
from . import views 

urlpatterns =[
    path('embedding/' , views.receive_embedding )
]