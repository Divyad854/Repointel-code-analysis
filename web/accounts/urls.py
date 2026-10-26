from django.contrib.auth import views as auth_views
from django.urls import path

from . import views
from .forms import UsernameOrEmailAuthenticationForm

urlpatterns = [
    path("register/", views.register, name="register"),
    path("register/verify/", views.verify_registration, name="verify_registration"),
    path("login/", auth_views.LoginView.as_view(
        template_name="registration/login.html",
        authentication_form=UsernameOrEmailAuthenticationForm,
        redirect_authenticated_user=True,
    ), name="login"),
    path("password/forgot/", views.forgot_password, name="password_reset"),
    path("password/forgot/verify/", views.verify_password_reset, name="password_reset_verify"),
    path("password/forgot/new/", views.set_reset_password, name="password_reset_set"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("profile/", views.profile, name="profile"),
    path("settings/password/", views.AccountPasswordChangeView.as_view(), name="password_change"),
    path("settings/github/connect/", views.connect_github, name="github_connect"),
    path("settings/github/callback/", views.github_callback, name="github_callback"),
]
