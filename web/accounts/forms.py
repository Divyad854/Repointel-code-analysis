from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import AuthenticationForm, SetPasswordForm, UserCreationForm
from django.core.validators import RegexValidator

User = get_user_model()

otp_validator = RegexValidator(r"^\d{6}$", "Enter the 6-digit code sent to your email.")


class EmailForm(forms.Form):
    email = forms.EmailField(label="Email address")

    def clean_email(self):
        return self.cleaned_data["email"].strip().lower()


class OTPForm(forms.Form):
    code = forms.CharField(
        label="Email verification code",
        max_length=6,
        validators=[otp_validator],
        widget=forms.TextInput(attrs={"inputmode": "numeric", "autocomplete": "one-time-code"}),
    )


class UsernameOrEmailAuthenticationForm(AuthenticationForm):
    username = forms.CharField(
        label="Username or email",
        widget=forms.TextInput(attrs={"autofocus": True, "autocomplete": "username"}),
    )

    def clean(self):
        identifier = self.cleaned_data.get("username", "").strip()
        if identifier and "@" in identifier:
            matching_users = User._default_manager.filter(email__iexact=identifier)
            if matching_users.count() == 1:
                self.cleaned_data["username"] = getattr(
                    matching_users.first(), User.USERNAME_FIELD
                )
        return super().clean()


class RegisterForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ("username", "email")

    def clean_email(self):
        email = self.cleaned_data["email"].strip().lower()
        if User.objects.filter(email__iexact=email).exists():
            raise forms.ValidationError("An account with this email already exists.")
        return email


class ProfileForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ("first_name", "last_name", "email")

    def clean_email(self):
        email = self.cleaned_data["email"].strip().lower()
        if User.objects.filter(email__iexact=email).exclude(pk=self.instance.pk).exists():
            raise forms.ValidationError("Another account already uses this email.")
        return email


class ResetPasswordForm(SetPasswordForm):
    pass
