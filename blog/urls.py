from django.urls import path
from .views import BlogListView,BlogDetailView,BlogCreateView,BlogUpdateView,BlogDeleteView,CommentCreateView,CommentDeleteView
from .views import AddVoteView,DeleteVoteView,LikeView,LikeHomeView,SummaryListView,PortfolioView
from .views import GenerateSummaryPdf, GenerateStockPdf
from django.urls import reverse_lazy
from wkhtmltopdf.views import PDFTemplateView

urlpatterns = [
    path('post/<int:pk>/delete/',BlogDeleteView.as_view(),name='post_delete'),
    path('post/<int:pk>/edit/',BlogUpdateView.as_view(),name='post_edit'),
    path('post/new/',BlogCreateView.as_view(),name='post_new'),
    path('post/<int:pk>/',BlogDetailView.as_view(),name='post_detail'),
    path('',BlogListView.as_view(),name='home'),
    path('posts', BlogListView.as_view(), name='all'),
    path('post/<int:pk>/comment',CommentCreateView.as_view(), name='post_comment_create'),
    path('comment/<int:pk>/delete',CommentDeleteView.as_view(), name='post_comment_delete'),
    path('post/<int:pk>/vote',AddVoteView.as_view(), name='post_vote'),
    path('post/<int:pk>/unvote',DeleteVoteView.as_view(), name='post_unvote'),
    path('like/<int:pk>',LikeView, name='like_post'),
    path('likehome/<int:pk>',LikeHomeView, name='like_home_post'),
    path('summary',SummaryListView.as_view(), name='summary_posts'),
    path('portfolio',PortfolioView.as_view(), name='portfolio_view'),
    path('pdf/', GenerateSummaryPdf.as_view()),
    path('pdf/<int:pk>', GenerateStockPdf.as_view()),
]
