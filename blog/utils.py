from io import BytesIO #A stream implementation using an in-memory bytes buffer
                       # It inherits BufferIOBase

from django.http import HttpResponse
from django.template.loader import get_template

#pisa is a html2pdf converter using the ReportLab Toolkit,
#the HTML5lib and pyPdf.

from xhtml2pdf import pisa
from datetime import date,timedelta
#difine render_to_pdf() function

def render_to_pdf(template_src, context_dict={}):
     template = get_template(template_src)
     html  = template.render(context_dict)
     result = BytesIO()

     pdf = pisa.pisaDocument(BytesIO(html.encode("UTF-8",errors='ignore')), result)
     if not pdf.err:
         return HttpResponse(result.getvalue(), content_type='application/pdf')
     return None
