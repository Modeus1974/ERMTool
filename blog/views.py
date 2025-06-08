from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponse,HttpResponseRedirect
from django.shortcuts import render, redirect, get_object_or_404
from django.views.generic import ListView, DetailView
from django.views.generic.edit import CreateView, UpdateView, DeleteView
from django.views import View
from django.urls import reverse,reverse_lazy
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.contrib.humanize.templatetags.humanize import naturaltime
from django.db.models import Q, Avg

from .models import Post,Comment,Vote
from .forms import CommentForm
from .finance import get_current_price,get_stock_stats,get_portfolio_stats
from .regime import return_regime_graph,return_portfolio_regime_graph
from .ta import TABacktester

import matplotlib.pyplot as plt
import io
import urllib, base64
from io import StringIO
import numpy as np
import scipy as sp
import pandas_datareader.data as getData
import yfinance as yf
from datetime import date,timedelta


def return_regime(stock):

    if type(stock) is list:
      plt,df = return_portfolio_regime_graph(stock)
    else:
      stock = stock + ""
      plt,df = return_regime_graph(stock)

    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    data = urllib.parse.quote(string)
    return data,df


def return_graph(stock):
    stock = stock + ""
    startdate  = date.today()-timedelta(365)
    enddate = date.today()
    consolidated = getData.DataReader(stock,data_source='yahoo',start=startdate,end=enddate)
    consolidated = consolidated.drop(['High','Low','Open','Close','Volume'],axis=1)

    fig = plt.figure()

    plt.title("Adjusted Closing Prices of " + stock + " over the past year")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Closing Price")
    consolidated["Adj Close"].plot(grid=True,label=stock)
    plt.legend()

    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    data = urllib.parse.quote(string)
    return data

def return_ta_graph(stock):
    stock = stock + ""
    print(stock)
    startdate  = date.today()-timedelta(365)
    enddate = date.today()

    smabt = TABacktester(stock,42,252,14,14,startdate, enddate,0)
    smabt.optimize()

    smabt.plot_SMA_results()
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    data_sma = urllib.parse.quote(string)

    smabt.plot_RSI_results()
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    data_rsi = urllib.parse.quote(string)

    smabt.plot_MR_results()
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    data_mr = urllib.parse.quote(string)

    return data_sma, data_rsi, data_mr

def return_stock_summary(stock):
    stock = stock + ""
    summary_stats = get_stock_stats(stock,5)
    return summary_stats


def LikeView(request,pk):
    post = get_object_or_404(Post, id=request.POST.get('post_id'))
    if post.likes.filter(id=request.user.id).exists():
        post.likes.remove(request.user)
        liked=False
    else:
        post.likes.add(request.user)
        liked=True
    return redirect(reverse('post_detail', args=[pk]))

def LikeHomeView(request,pk):
    post = get_object_or_404(Post, id=request.POST.get('post_id'))
    if post.likes.filter(id=request.user.id).exists():
        post.likes.remove(request.user)
        liked=False
    else:
        post.likes.add(request.user)
        liked=True
    return redirect(reverse('home'))

class SummaryListView(ListView):
    model = Post
    template_name = 'summary.html'

    def get(self, request):
        x = Post.objects.order_by('-include')
        average = x.filter(include=True).aggregate(Avg('projected_yield'))
        context = { 'sorted_posts' : x, 'future_yield' : average }
        return render(request, self.template_name, context)

class PortfolioView(ListView):
    model = Post
    template_name = 'portfolio.html'

    def get(self, request):
        x = Post.objects.order_by('-include')
        stock_list = x.filter(include=True)
        stock_index = []
        for i in stock_list:
          stock_index.append(i.ticker)

        average = stock_list.aggregate(Avg('projected_yield'))
        context = { 'sorted_posts' : stock_list, 'future_yield' : average }
        context['portfolio_summary'] = get_portfolio_stats(stock_index,5).to_html()
        context['graph'],df = return_regime(stock_index)
        context['regime_data'] = df.to_html()
        return render(request, self.template_name, context)

class BlogListView(ListView):
    model = Post
    template_name = 'home.html'

class BlogDetailView(DetailView):
    model = Post
    template_name = 'post_detail.html'
    def get(self, request, pk) :
        x = Post.objects.get(id=pk)
        comments = Comment.objects.filter(post=x).order_by('-updated_at')
        comment_form = CommentForm()
        stuff = get_object_or_404(Post, id=self.kwargs['pk'])
        total_likes = stuff.total_likes()
        if stuff.likes.filter(id=self.request.user.id).exists():
            liked=True
        else:
            liked=False
        context = { 'post' : x, 'comments': comments, 'comment_form': comment_form,'total_likes':total_likes,'liked':liked }
        context['graph'],df = return_regime(x.ticker)
        context['regime_data'] = df.to_html()
        #context['sma_graph'], context['rsi_graph'], context['mr_graph'] = return_ta_graph(x.ticker)
        context['latest_close'] = get_current_price(x.ticker)
        context['stock_summary'] = return_stock_summary(x.ticker).to_html()
        print(get_current_price(x.ticker))
        return render(request, self.template_name, context)

class BlogCreateView(CreateView):
    model = Post
    template_name = 'post_new.html'
    fields = ['title','ticker','current_price','target_price','current_yield','projected_yield','author','business','pros','cons','include','rationale']

class BlogUpdateView(UpdateView):
    model = Post
    template_name = 'post_edit.html'
    fields = ['title','ticker','current_price','target_price','current_yield','projected_yield','business','pros','cons','include','rationale']

class BlogDeleteView(DeleteView):
    model = Post
    template_name = 'post_delete.html'
    success_url = reverse_lazy('home')

class CommentCreateView(LoginRequiredMixin, View):
    def post(self, request, pk) :
        f = get_object_or_404(Post, id=pk)
        comment = Comment(text=request.POST.get('comment'), author=request.user, post=f)
        comment.save()
        return redirect(reverse('post_detail', args=[pk]))

class CommentDeleteView(DeleteView):
    model = Comment
    template_name = "comment_delete.html"

    def get_success_url(self):
        post = self.object.post
        return reverse('post_detail', args=[post.id])

from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.db.utils import IntegrityError

@method_decorator(csrf_exempt, name='dispatch')
class AddVoteView(LoginRequiredMixin, View):
    def post(self, request, pk) :
        print("Add PK",pk)
        t = get_object_or_404(Post, id=pk)
        vote = Vote(author=request.user, post=t)
        try:
            vote.save()  # In case of duplicate key
        except IntegrityError as e:
            pass
        return HttpResponse()

@method_decorator(csrf_exempt, name='dispatch')
class DeleteVoteView(LoginRequiredMixin, View):
    def post(self, request, pk) :
        print("Delete PK",pk)
        t = get_object_or_404(Post, id=pk)
        try:
            vote = Vote.objects.get(author=request.user, post=t).delete()
        except Vote.DoesNotExist as e:
            pass
        return HttpResponse()


#importing get_template from loader
from django.template.loader import get_template

#import render_to_pdf from util.py
from .utils import render_to_pdf

#Creating our view, it is a class based view
class GenerateSummaryPdf(ListView):
    model = Post
    template_name = 'pdf_detail.html'

    def get(self, request):
        x = Post.objects.order_by('-include')
        stock_list = x.filter(include=True)
        stock_index = []
        for i in stock_list:
          stock_index.append(i.ticker)

        average = stock_list.aggregate(Avg('projected_yield'))
        context = { 'sorted_posts' : stock_list, 'future_yield' : average }
        context['portfolio_summary'] = get_portfolio_stats(stock_index,5).to_html()
        context['graph'],df = return_regime(stock_index)
        context['regime_data'] = df.to_html()
        context['today_date'] = date.today()
        pdf = render_to_pdf(template_src = 'pdf_detail.html',context_dict=context)
         #rendering the template
        return HttpResponse(pdf, content_type='application/pdf')


class GenerateStockPdf(ListView):
    model = Post
    template_name = 'stock_detail_pdf.html'

    def get(self, request, pk, *args, **kwargs):
        x = Post.objects.get(id=pk)
        comments = Comment.objects.filter(post=x).order_by('-updated_at')
        comment_form = CommentForm()
        stuff = get_object_or_404(Post, id=self.kwargs['pk'])
        total_likes = stuff.total_likes()
        if stuff.likes.filter(id=self.request.user.id).exists():
            liked=True
        else:
            liked=False
        context = { 'post' : x, 'comments': comments, 'comment_form': comment_form,'total_likes':total_likes,'liked':liked }
        context['graph'],df = return_regime(x.ticker)
        context['regime_data'] = df.to_html()
        context['latest_close'] = get_current_price(x.ticker)
        context['stock_summary'] = return_stock_summary(x.ticker).to_html(header=False)
        #context['sma_graph'], context['rsi_graph'], context['mr_graph'] = return_ta_graph(x.ticker)
        #print(context['stock_summary'])
        pdf = render_to_pdf(template_src = 'stock_detail_pdf.html',context_dict=context)
         #rendering the template
        return HttpResponse(pdf, content_type='application/pdf')
