from django.urls import path
from Sign_Application import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns=[
    path('hand_signal_detection/',views.hand_signal_detection,name="hand_signal_detection"),
    path('home',views.Home,name="Home"),
    path('video-feed/', views.video_feed, name='video_feed'),
    path('show_images/',views.show_imagess,name="show_images"),
    path('Registration_save/',views.Registration_save,name="Registration_save"),

    path('login/',views.Login_Pg,name="Login_Pg"),
    path('',views.MainHomeView.as_view(),name="main"),
    path('RegistrationForm/',views.RegistrationForm,name="RegistrationForm"),

    path('Login_fun/',views.Login_fun,name="Login_fun"),
    path('Logout_fn/',views.Logout_fn,name="Logout_fn"),
    # path('shows/',views.shows,name='s'),
    path('learn/',views.Learning_page.as_view(),name="learn"),
    path('video/',views.Live.as_view(),name='video'),
    path('text_to_sign/',views.Text_to_SignLanguage.as_view(),name="text_to_sign"),
    path('forget-password/', views.forgetpassword_enteremail, name="forget_password_enter_email"),
    path('otp',views.otp,name='otp'),
    path('new_password',views.newpassword,name='new_password'),
    path('get_latest_data/', views.get_latest_data, name='get_latest_data'),
    path('add_audio/', views.add_audio, name='add_audio'),
    path('command/', views.handle_command, name='command'),
    path('text-to-voice/', views.TextToVoiceView.as_view(), name='text-voice'),
    path('voice-clone/', views.voice_clone_view, name='voice-clone'),


]+ static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
