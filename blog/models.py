from django.db import models
from django.urls import reverse
from django.utils import timezone
from django.core.validators import MinLengthValidator

class Post(models.Model):
    title=models.CharField(max_length=50,default="")
    ticker=models.CharField(max_length=10,default="")
    current_price=models.FloatField(default=0)
    target_price=models.FloatField(default=0)
    current_yield=models.FloatField(default=0)
    projected_yield=models.FloatField(default=0)
    author=models.ForeignKey('auth.User', on_delete=models.CASCADE,)
    business=models.TextField(max_length=500,default="")
    pros = models.TextField(max_length=500,default="")
    cons = models.TextField(max_length=500,default="")
    include=models.BooleanField(default=True)
    rationale=models.TextField(max_length=500,default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    likes = models.ManyToManyField('auth.user', related_name="blog_posts")
    votes = models.ManyToManyField('auth.user', through='Vote', related_name='voted_posts')

    def total_likes(self):
        return self.likes.count()

    def __str__(self):
        return self.title

    def get_absolute_url(self):
        return reverse('post_detail', args=[str(self.id)])

class Comment(models.Model) :
    text = models.TextField(
        validators=[MinLengthValidator(3, "Comment must be greater than 3 characters")]
    )

    post = models.ForeignKey(Post, on_delete=models.CASCADE)
    author = models.ForeignKey('auth.User', on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Shows up in the admin list
    def __str__(self):
        if len(self.text) < 15 : return self.text
        return self.text[:11] + ' ...'

class Vote(models.Model) :
    post = models.ForeignKey(Post, on_delete=models.CASCADE)
    author = models.ForeignKey('auth.User', on_delete=models.CASCADE)

    class Meta:
        unique_together = ('post', 'author')

    def __str__(self) :
        return '%s likes %s'%(self.author.username, self.post.ticker)
