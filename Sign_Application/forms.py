from django import forms

class SpeechForm(forms.Form):
    user_input = forms.TextInput()
